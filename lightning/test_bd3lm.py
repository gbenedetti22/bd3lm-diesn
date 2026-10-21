"""
Tests of the DiESN backbone for BD3LM and of the Lightning module (GPU: the ParalESN scan is a Triton kernel).

    uv run pytest -q lightning/test_bd3lm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: diffusion/, models/, paralesn/, data/

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diesn_bd3 import DiESNBD3  # noqa: E402
from diffusion.config import DiESNConfig  # noqa: E402
from diffusion.diesn import DIESN  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

V, BS, L = 50258, 16, 64
MASK = V - 1
TINY = ["model.length=64", "block_size=16", "model.hidden_size=64", "model.n_blocks=2", "model.n_heads=2",
        "model.cond_dim=32", "eval.gen_ppl_eval_model_name_or_path=gpt2"]


def _config(*overrides):
    base = OmegaConf.load(Path(__file__).with_name("config.yaml"))
    return OmegaConf.merge(base, OmegaConf.from_dotlist(TINY + list(overrides)))


def _randomize(model):
    """Non-zero AdaLN gates and output head (at init the blocks contribute nothing and the logits are 0)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "adaLN" in n or "output_layer.linear" in n:
                p.normal_(0, 0.05)
    return model


def _backbone(seed=0):
    torch.manual_seed(seed)
    return _randomize(DiESNBD3(_config(), V).cuda().eval())


def _reference_diesn():
    cfg = DiESNConfig(hidden_size=64, n_blocks=2, cond_dim=32, dropout=0.1, esn_n_units=64, esn_n_heads=2,
                      attn_impl="softmax")
    return DIESN(OmegaConf.structured(cfg), V)


def _close(a, b, tol=0.01):  # bf16: ~1e-3 when equal, ~0.1 with a wrong context
    a, b = a.float(), b.float()
    return (a - b).norm() <= tol * a.norm()


def _data(B=2):
    x0 = torch.randint(0, V - 2, (B, L), device="cuda")
    xt = torch.where(torch.rand(B, L, device="cuda") < 0.5, MASK, x0)
    return x0, xt, torch.zeros(B, device="cuda")


# ------------------------------------------------------------------------------------------------------------------
# Backbone

def test_state_dict_is_the_diesn_one():
    m = _backbone()
    ref = _reference_diesn()
    assert m.net.state_dict().keys() == ref.state_dict().keys()
    ref.load_state_dict(m.net.state_dict())  # strict


def test_inside_a_block_it_is_the_bidirectional_diesn():
    """Within one block block_diff_mask lets every position read the whole block: the logits are those of the
    original DIESN (bidirectional ParalESN of paralesn, Conv1d, softmax attention)."""
    m = _backbone()
    ref = _reference_diesn().cuda().eval()
    ref.load_state_dict(m.net.state_dict())
    x, sigma = torch.randint(0, V - 2, (3, BS), device="cuda"), torch.zeros(3, device="cuda")
    with torch.no_grad():
        assert _close(ref(x, sigma), m(x, sigma, sample_mode=True))


def test_esn_is_the_reference_reservoir():
    """x0 half of the ParalESN = paralesn reservoir with the forward scan over the whole sequence and the backward
    scan inside every block, followed by the reference mixer."""
    m = _backbone()
    esn = m.net.blocks[0].esn
    reservoir, mixer = esn.layers[0].reservoir, esn.layers[0].mixer
    B, D = 2, 64
    x = torch.randn(B, L, D, device="cuda")
    esn.half = "x0"
    with torch.no_grad():
        out = esn(x)
        xc = x.to(torch.complex64)
        fwd = reservoir(xc)
        bwd = reservoir(xc.reshape(B * L // BS, BS, D), bidirectional=True).reshape(B, L, -1)[..., reservoir.n_units:]
        ref = mixer(torch.cat([fwd, bwd], dim=-1))
    assert _close(ref, out, tol=1e-3)


def test_training_halves_match_sampling():
    """Training (forward on [x_t, x0]) gives block b the logits that sampling gives it with one pass on
    [x0 blocks < b, x_t block b] (sample_mode, as DIT)."""
    m = _backbone()
    x0, xt, sigma = _data()
    with torch.no_grad():
        train = m(torch.cat([xt, x0], dim=1), sigma)
        for b in range(L // BS):
            seq = torch.cat([x0[:, :b * BS], xt[:, b * BS:(b + 1) * BS]], dim=1)
            assert _close(m(seq, sigma, sample_mode=True, last=BS), train[:, b * BS:(b + 1) * BS]), f"block {b}"


def test_xt_block_does_not_see_its_target():
    """The logits of the x_t block b do not depend on x0 at blocks >= b."""
    m = _backbone()
    x0, xt, sigma = _data()
    b = 2
    other = x0.clone()
    other[:, b * BS:] = torch.randint(0, V - 2, (2, L - b * BS), device="cuda")
    with torch.no_grad():
        a = m(torch.cat([xt, x0], dim=1), sigma)[:, :(b + 1) * BS]
        c = m(torch.cat([xt, other], dim=1), sigma)[:, :(b + 1) * BS]
    assert torch.equal(a, c)


def test_x0_context_matters():
    """With the x0 of another sequence the logits of blocks > 0 change (the test above is not trivially true)."""
    m = _backbone()
    x0, xt, sigma = _data()
    other = torch.randint(0, V - 2, x0.shape, device="cuda")
    with torch.no_grad():
        a, c = m(torch.cat([xt, x0], dim=1), sigma), m(torch.cat([xt, other], dim=1), sigma)
    assert torch.equal(a[:, :BS], c[:, :BS]) and not _close(a[:, BS:], c[:, BS:], tol=0.05)


def test_one_hot_soft_masks_are_the_tokens():
    m = _backbone()
    x0, _, sigma = _data()
    soft = (F.pad(x0[..., None], (0, 3)), F.pad(torch.ones(*x0.shape, 1, device="cuda"), (0, 3)))
    with torch.no_grad():
        assert torch.equal(m(x0, sigma, sample_mode=True), m(soft, sigma, sample_mode=True))


# ------------------------------------------------------------------------------------------------------------------
# Lightning module (diffusion.py ported)

def _bd3lm(*overrides):
    from bd3lm import BD3LM

    torch.manual_seed(0)
    return BD3LM(_config("training.ema=0", "model.attn_backend=sdpa", *overrides)).to("cuda")  # to(), as Lightning


@pytest.mark.parametrize("overrides", [["algo.backbone=dit"], ["algo.soft_masking=none"],
                                       ["algo.soft_masking=smdm", "optim.sm_prob=1.0"]])
def test_every_parameter_gets_a_gradient(overrides):
    """Every trainable parameter gets a gradient (the TransparencyHead's temperature, unused by its forward, is
    frozen; with soft masking in every step, sm_prob=1)."""
    m = _bd3lm(*overrides).train()
    x0, _, _ = _data()
    losses = m._loss(x0, torch.ones_like(x0))
    loss = losses.loss
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_soft_masks_do_not_see_the_target():
    """With soft masking the logits of the x_t block b still do not depend on x0 at blocks >= b: the NELBO stays a
    bound."""
    m = _randomize(_bd3lm("algo.soft_masking=smdm")).eval()
    with torch.no_grad():  # make the heads mix: lambda = scale sigmoid(steepness (-entropy - centre)) -> 1, with the
        for head in m._soft_heads():  # near-uniform predictions of a random model (entropy ~ log V ~ 10.8)
            head.raw_scale.fill_(5.0)
            head.raw_centre_neg.fill_(20.0)  # centre ~ -20
    x0, xt, _ = _data()
    b = 2
    other = x0.clone()
    other[:, b * BS:] = torch.randint(0, V - 2, (2, L - b * BS), device="cuda")
    sigma = torch.zeros(2, 1, device="cuda")
    with torch.no_grad():
        a = m._soft_masked_logits(xt, x0, sigma)[:, :(b + 1) * BS]
        c = m._soft_masked_logits(xt, other, sigma)[:, :(b + 1) * BS]
        binary = m._backbone_logits(torch.cat([xt, x0], dim=1), sigma)[:, :(b + 1) * BS]
    assert torch.equal(a, c)
    assert not torch.equal(a, binary)  # the soft masks do change the prediction


def test_soft_masking_probability():
    """As SMDM: a training step uses soft masks with probability optim.sm_prob (binary masks otherwise, where the
    head gets no gradient: train.py runs DDP with find_unused_parameters), evaluation always."""
    m = _bd3lm("algo.soft_masking=smdm", "optim.sm_prob=0.0").train()
    x0, _, _ = _data()
    m._loss(x0, torch.ones_like(x0)).loss.backward()
    assert m.soft_mask.raw_scale.grad is None
    calls = []
    m.soft_mask.register_forward_hook(lambda *args: calls.append(1))
    m.eval()
    with torch.no_grad():
        m._loss(x0, torch.ones_like(x0))
    assert calls


def test_subs_log_p_theta_is_the_bd3lm_one():
    """The training loss reads log p_theta(x0 | x_t) on the masked tokens only: the value of
    gather(_subs_parameterization(logits, x_t), x0) of diffusion.py."""
    m = _bd3lm()
    x0, xt, _ = _data()
    logits = torch.randn(*x0.shape, V, device="cuda")
    ref = torch.gather(m._subs_parameterization(logits.clone(), xt), -1, x0[..., None]).squeeze(-1)
    assert torch.allclose(m._subs_log_p_theta(logits.clone(), xt, x0), ref, atol=1e-4)


def test_compiled_backbone_matches_eager():
    """compile_backbone (torch.compile of the per-token computation, fused depthwise convolution) computes the same
    logits as the eager backbone."""
    eager, compiled = _bd3lm().eval(), _bd3lm().eval()  # same seed: same weights and reservoirs
    _randomize(eager), compiled.load_state_dict(eager.state_dict())
    compiled.compile_backbone()
    x0, xt, _ = _data()
    sigma = torch.zeros(2, 1, device="cuda")
    with torch.no_grad():
        a = eager._backbone_logits(torch.cat([xt, x0], dim=1), sigma)
        b = compiled._backbone_logits(torch.cat([xt, x0], dim=1), sigma)
    assert _close(a, b)


def test_clipped_schedule_resampling():
    """q_xt with resample: the masked fraction of every block inside [eps_min, eps_max]."""
    m = _bd3lm()
    x0, _, _ = _data(8)
    t = m._sample_t(x0.shape, x0.device, 0.3, 0.8)
    xt = m.q_xt(x0, m.noise(t)[1], sampling_eps_min=0.3, sampling_eps_max=0.8)
    frac = (xt == m.mask_index).float().view(8, -1, BS).mean(-1)
    assert ((frac >= 0.3) & (frac <= 0.8)).all()


@pytest.mark.parametrize("overrides", [["algo.backbone=dit"], ["algo.soft_masking=none"],
                                       ["algo.soft_masking=smdm"]])
@pytest.mark.parametrize("first_hitting", [True, False])
def test_sampling(overrides, first_hitting):
    m = _bd3lm(*overrides, f"sampling.first_hitting={first_hitting}", "loader.eval_batch_size=3").eval()
    num_steps = 1000 if first_hitting else 4
    samples, nfes = m._semi_ar_sampler(n_samples=3, num_steps=num_steps, num_strides=L // BS, seqlen=L)
    if samples is None:  # rejected by _check_stop_conds: only for sequences longer than 256 tokens
        pytest.fail("rejected")
    assert samples.shape == (3, L) and (samples[:, 0] == m.bos_token_id).all() and not (samples == MASK).any()
    assert len(m.restore_model_and_sample(num_steps=num_steps)) == 3
