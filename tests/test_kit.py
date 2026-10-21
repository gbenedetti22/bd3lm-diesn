import copy

import pytest
import torch
from omegaconf import OmegaConf

from diffusion.config import DiESNConfig
from diffusion.diesn import DIESN
from diffusion.transparency_head import TransparencyHead
from models.fast_esn import use_fast_esn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_imports():
    import data.datamodule  # noqa: F401
    import data.openwebtext  # noqa: F401
    from dataset import InMemoryDataset, StreamingDataset  # noqa: F401
    from dataset.openwebtext import OpenWebTextDataset  # noqa: F401
    TransparencyHead(mask_token_id=50257)


@cuda
@pytest.mark.parametrize("attn_impl", ["gla", "softmax"])
def test_diesn_fast_esn_matches_reference(attn_impl):
    torch.manual_seed(0)
    cfg = OmegaConf.structured(DiESNConfig(hidden_size=64, n_blocks=2, cond_dim=32, esn_n_units=64, esn_n_heads=2,
                                           attn_impl=attn_impl))
    ref = DIESN(cfg, vocab_size=50258).cuda().eval()
    with torch.no_grad():  # non-zero AdaLN gates and output head (at init the blocks contribute nothing)
        for n, p in ref.named_parameters():
            if "adaLN" in n or "output_layer.linear" in n:
                p.normal_(0, 0.05)
    fast = copy.deepcopy(ref)
    assert use_fast_esn(fast) == 2
    x, sigma = torch.randint(0, 50257, (2, 128), device="cuda"), torch.rand(2, device="cuda")
    with torch.no_grad():
        a, b = ref(x, sigma).float(), fast(x, sigma).float()
    assert a.shape == (2, 128, 50258)
    assert (a - b).norm() < 0.02 * a.norm()
