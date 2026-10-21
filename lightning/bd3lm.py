"""
BD3LM (Block Diffusion, arXiv:2503.09573) as a PyTorch Lightning module: the class Diffusion of diffusion.py with
algo=bd3lm. Its code paths are kept as they are; the other ones are removed (AR, SEDD, MDLM, analytic sampler, KV
cache, HF backbones, generative perplexity, Hydra). Additions and changes are marked "lightning:":
  - the DiESN backbone (algo.backbone=diesn, lightning/diesn_bd3.py) in place of the Transformer (algo.backbone=dit,
    models/dit.py, kept as baseline);
  - soft masking (algo.soft_masking, DiESN only) with the TransparencyHead of diffusion/transparency_head.py.

Soft masking (algo.soft_masking=smdm, _soft_masked_logits), as in SMDM (IBM/soft-masked-diffusion-language-models,
language/algo.py: MDLM_SM), inside the forward on [x_t, x0]. The soft mask of a masked token mixes [MASK] with a
preliminary prediction p of it: (1 - lambda) embed([MASK]) + lambda sum_k p_k embed(top-k token), lambda from the
entropy of p (TransparencyHead). Training: the x0 half (the forward on the clean sequence that BD3LM adds to
training) runs once and its context serves both passes on the noisy blocks: a preliminary x_t pass on binary masks,
without gradient, gives p; the x_t pass on the soft masks gives the output. Soft masking thus costs one x_t pass
without gradient, not a second BD3LM forward (x0 and x_t). As in SMDM it is used in a training step with probability
optim.sm_prob (binary masks otherwise), always in evaluation; the head has its own learning rate
(optim.tran_head_lr) and initialization (algo.tran_head). Sampling: binary masks at the first step of a block, then p
from the prediction of the previous step. The masks of block b depend only on x_t^b and on x^{<b}: the NELBO stays
a bound of the likelihood.
"""

import itertools
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from omegaconf import OmegaConf
from tqdm import tqdm

import metrics
import noise_schedule
from data.config import DataConfig
from data.datamodule import TokenDataModule
from data.tokenizer import Tokenizer
from diesn_bd3 import DiESNBD3
from diffusion.transparency_head import TransparencyHead
from models.dit import DIT
from models.ema import ExponentialMovingAverage


def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class BD3LM(L.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    # lightning: the GPT-2 tokenizer of the dataset (data/tokenizer.py); [MASK] = 50257 after the GPT-2 vocabulary,
    # as diffusion.py does for tokenizers without a mask token
    self.tokenizer = Tokenizer()
    self.vocab_size = self.tokenizer.refiner_vocab_size
    self.mask_index = self.tokenizer.mask_token_id
    self.bos_token_id = self.eos_token_id = self.tokenizer.eot_token_id  # <|endoftext|>: GPT-2's BOS and EOS
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    self.parameterization = self.config.algo.parameterization
    self.block_size = self.config.block_size
    if self.config.algo.backbone == 'dit':
      # lightning: built with the SDPA mask (on CPU); the training mask (model.attn_backend) is built on the
      # module's device in to(), not on cuda:0 in every DDP process
      dit_config = OmegaConf.merge(self.config, {'model': {'attn_backend': 'sdpa'}})
      self.backbone = DIT(dit_config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'diesn':
      self.backbone = DiESNBD3(self.config, vocab_size=self.vocab_size)  # lightning
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')
    self._dit_mask = None

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    # lightning: soft masking head, initialized as SMDM (algo.tran_head). With transparency_alg mixinputs_with_topk,
    # the one SMDM trains with, the temperature is not used (TransparencyHead.forward, also in SMDM's original): it
    # would never get a gradient, so it is frozen (same value, same state_dict)
    self.soft_masking = self.config.algo.soft_masking
    self.soft_mask = None
    if self.soft_masking == 'smdm':
      self.soft_mask = TransparencyHead(self.mask_index, **OmegaConf.to_container(self.config.algo.tran_head))
      self.soft_mask.raw_temperature.requires_grad_(False)

    if self.config.training.ema > 0:
      self.ema = ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None

    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))

    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.train_time_s = 0.0  # lightning: training time (train.py: TrainTimer), kept in the checkpoints
    self._validate_configuration()

  def _soft_heads(self):
    return [self.soft_mask] if self.soft_mask is not None else []

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    parameters += [head.parameters() for head in self._soft_heads()]  # lightning
    return itertools.chain(* parameters)

  def _validate_configuration(self):
    assert self.config.algo.name == 'bd3lm' and self.parameterization == 'subs'
    assert self.config.algo.backbone in {'dit', 'diesn'}
    assert not self.config.sampling.kv_cache, 'KV cache: not ported'
    assert not self.time_conditioning
    assert self.soft_masking in {'none', 'smdm'}
    if self.soft_masking != 'none':
      assert self.config.algo.backbone == 'diesn', 'soft masks are fed to the embedding of diesn.py'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'

  def compile_backbone(self):
    """lightning: torch.compile (in place: same modules and state_dict keys) of the per-token computation of the
    backbone blocks, for training: it fuses norms, AdaLN modulation, residuals, dropout and activations, which run
    as separate memory-bound kernels otherwise. DiESN: the mixers under block_diff_mask stay eager (diesn_bd3.py)."""
    if self.config.algo.backbone == 'diesn':
      self.backbone.compile_blocks()
    else:
      for module in [*self.backbone.blocks, self.backbone.output_layer]:
        module.compile()

  def _dit_attention(self, attn_backend):
    """lightning: (re)build the block-diffusion mask of models/dit.py for `attn_backend` on this module's device
    (DIT keeps it as a plain attribute, not a buffer)."""
    if self._dit_mask == (attn_backend, self.device):
      return
    self.backbone.attn_backend = attn_backend
    for block in self.backbone.blocks:
      block.attn_backend = attn_backend
    with torch.cuda.device(self.device):
      self.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend=attn_backend)
    if attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    self._dit_mask = (attn_backend, self.device)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.metrics.to(*args, **kwargs)
    if self.config.algo.backbone == 'dit' and self.device.type == 'cuda':
      self._dit_attention(self.config.model.attn_backend)  # lightning
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    self.train_time_s = checkpoint.get('train_time_s', 0.0)  # lightning

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    checkpoint['train_time_s'] = self.train_time_s  # lightning

  def _ema_to_device(self):
    """lightning: the EMA weights on the module's device (loaded on CPU with the checkpoint)."""
    if self.ema and self.ema.shadow_params and self.ema.shadow_params[0].device != self.device:
      self.ema.move_shadow_params_to_device(self.device)

  def on_train_start(self):
    # lightning: Lightning's (distributed) samplers instead of the fault-tolerant ones of dataloader.py
    self._ema_to_device()

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity

    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _backbone_logits(self, x, sigma, sample_mode=False, store_kv=False, soft=None, last_block=False):
    """lightning: forward() before the SUBS parameterization. soft, the soft masks (indices, probs) given to the
    backbone in place of x; last_block, the logits of the last block only (batched sampling: the (n, T, V) logits
    in float64 of diffusion.py do not fit)."""
    sigma = self._process_sigma(sigma)
    kwargs = {'last': self.block_size} if last_block and self.config.algo.backbone == 'diesn' else {}
    # diffusion.py: autocast(dtype=float32), which on CUDA disables autocast; the backbones autocast to bf16 inside
    with torch.amp.autocast('cuda', enabled=False):
      logits = self.backbone(x if soft is None else soft, sigma,
                             store_kv=store_kv,
                             sample_mode=sample_mode, **kwargs)
    return logits[:, -self.block_size:] if last_block else logits

  def forward(self, x, sigma, sample_mode=False, store_kv=False, soft=None, last_block=False):
    """Returns log score."""
    logits = self._backbone_logits(x, sigma, sample_mode, store_kv, soft, last_block)
    if self.cross_attn:
      x = x[:, :self.config.model.length]
    if last_block:
      x = x[:, -self.block_size:]
    return self._subs_parameterization(logits=logits, xt=x)

  def _subs_log_p_theta(self, logits, xt, x0):
    """lightning: log p_theta(x0 | x_t) of the SUBS parameterization, the value of
    gather(_subs_parameterization(logits, xt), x0) in _forward_pass_diffusion of diffusion.py: log-softmax over the
    vocabulary without [MASK] at the masked tokens, 0 at the others (copied from x_t). Computed on the masked tokens
    only: _subs_parameterization normalizes and rewrites the whole (B, L, V) logits, ~10% of a training step."""
    masked = xt == self.mask_index
    logits = logits[masked]
    logits[:, self.mask_index] += self.neg_infinity
    log_p = torch.zeros(xt.shape, device=xt.device)
    log_p[masked] = -F.cross_entropy(logits, x0[masked], reduction='none').float()
    return log_p

  # lightning: soft masking ------------------------------------------------------------------------------------

  def _no_mask_logit(self, logits):
    """The prediction never puts [MASK] in the top-k (SUBS: the denoiser does not predict it)."""
    index = torch.tensor([self.mask_index], device=logits.device)
    return logits.index_fill(-1, index, self.neg_infinity)

  @staticmethod
  def _one_hot(x):
    """(indices, probs) of the tokens themselves: binary masks."""
    return x[..., None], torch.ones(*x.shape, 1, device=x.device)

  @staticmethod
  def _cat_positions(a, b):
    """Soft inputs a, b concatenated along the sequence (the narrower one padded with zero-weight entries)."""
    width = max(a[0].shape[-1], b[0].shape[-1])
    pad = lambda t: F.pad(t, (0, width - t.shape[-1]))  # noqa: E731
    return torch.cat([pad(a[0]), pad(b[0])], dim=1), torch.cat([pad(a[1]), pad(b[1])], dim=1)

  def _soft_masked_logits(self, xt, x0, sigma):
    """BD3LM's forward on [x_t, x0] with SMDM soft masks at the masked tokens of x_t (module docstring): logits of
    the x_t half."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', enabled=False):
      self.backbone.x0_pass(x0, sigma)  # the x0 half, once: its context serves both x_t passes
      with torch.no_grad():
        prelim = self._no_mask_logit(self.backbone.xt_pass(xt, sigma))  # binary masks
      # the bf16 copies of the weights cast without gradient above stay in the autocast cache until the outermost
      # autocast context exits: the pass below would reuse them and give their parameters no gradient
      torch.clear_autocast_cache()
      return self.backbone.xt_pass(self.soft_mask(xt, prelim), sigma)

  def _sampling_soft_masks(self, x, prev_log_p):
    """Soft masks of x: one-hot clean blocks, then the block being denoised with SMDM soft masks from the
    log-probabilities of the previous step."""
    block = x[:, -self.block_size:]
    return self._cat_positions(self._one_hot(x[:, :-self.block_size]), self.soft_mask(block, prev_log_p.float()))

  # --------------------------------------------------------------------------------------------------------------

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    # lightning: the batches of data/datamodule.py are token chunks (B, L), without padding
    losses = self._loss(batch, torch.ones_like(batch))
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    if self.soft_mask is not None:  # lightning: the transparency parameters, as SMDM's MDLM_SM.training_step
      for name in ('scale', 'centre', 'steepness'):
        self.log(f'transparency/{name}', getattr(self.soft_mask, name).item(), on_step=True, on_epoch=False,
                 sync_dist=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self._ema_to_device()  # lightning
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)

  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)

  def validation_step(self, batch, batch_idx):
    batch = {'input_ids': batch, 'attention_mask': torch.ones_like(batch)}  # lightning: see training_step
    if self.var_min:
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # only report variance over 100 batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # lightning: the soft masking heads in their own group, with a higher learning rate (SMDM)
    groups = [{'params': itertools.chain(self.backbone.parameters(), self.noise.parameters())}]
    heads = [p for head in self._soft_heads() for p in head.parameters() if p.requires_grad]
    if heads:
      groups.append({'params': heads, 'lr': self.config.optim.tran_head_lr})
    optimizer = torch.optim.AdamW(
      groups,
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    # lightning: configs/lr_scheduler/constant_warmup.yaml without hydra.utils.instantiate
    scheduler = transformers.get_constant_schedule_with_warmup(
      optimizer, num_warmup_steps=self.config.lr_scheduler.num_warmup_steps)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt

  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size

    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None, soft=None):
    """lightning: soft, the soft masks of x (soft masking); also returns the log-probabilities of the block
    computed at this step (None if p_x0 was reused), for the soft masks of the next step."""
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    log_p = None
    if p_x0 is None:
      # lightning: the log score of the last block only (diffusion.py: of the whole sequence, then sliced)
      log_p = self.forward(x,
                           sigma_t,
                           sample_mode=True,
                           soft=soft,
                           last_block=True).to(torch.float64)
      p_x0 = log_p.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      # lightning: for every sequence of the batch (diffusion.py: torch.randint, batch size 1)
      masked = x[:, -self.block_size:] == self.mask_index
      ind = torch.where(masked, torch.rand(masked.shape, device=x.device), -1.).argmax(-1)
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    if not torch.allclose(x_new, x):
      return None, x_new, log_p
    else:
      return p_x0, x_new, log_p

  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.config.algo.backbone == 'dit':
      self._dit_attention('sdpa')  # lightning: models/dit.py samples with the SDPA mask only
    samples = []
    for _ in range(self.config.sampling.num_sample_batches):
      sample_i, num_tries = None, 0
      while sample_i is None:
        num_tries += 1
        sample_i, nfes = self._semi_ar_sampler(
          n_samples=batch_size_per_gpu,
          num_strides=(seqlen // self.block_size),
          num_steps=num_steps,
          seqlen=seqlen)
        if num_tries > 10:
          raise ValueError('Sampling failed.')
      samples.append(sample_i)
      self.metrics.nfes.update(nfes)
      self.metrics.gen_nfes.append(nfes)
    samples = torch.cat(samples, dim=0)
    return self.tokenizer.decode_batch(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:
      self._ema_to_device()  # lightning
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    # lightning: no generative perplexity (metrics.record_generative_perplexity)
    return samples

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    # lightning: the chunks of data/datamodule.py have the model length (diffusion.py also crops 2x sequences)
    assert x0.shape[1] == self.num_tokens
    return x0, None, attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]

    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    if self.config.algo.backbone == 'dit':  # lightning: the training mask (also after sampling or .cuda())
      self._dit_attention(self.config.model.attn_backend)
    # lightning: the backbone's logits, then the SUBS log-probability of x0 on the masked tokens only
    # (diffusion.py: model_output = self.forward(x_input, sigma) and log_p_theta = gather(model_output, x0))
    # soft masks in the forward on [x_t, x0]: as SMDM (MDLM_SM.nll), in a training step only with probability
    # optim.sm_prob, always in evaluation
    use_soft_mask = self.soft_masking != 'none' and (
      not self.training or torch.rand(1).item() < self.config.optim.sm_prob)
    if use_soft_mask:
      logits = self._soft_masked_logits(xt, x0, sigma)
    else:
      logits = self._backbone_logits(x_input, sigma=sigma)
    log_p_theta = self._subs_log_p_theta(logits, xt, x0)
    loss = loss_scale * log_p_theta
    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    loss = self._forward_pass_diffusion(
      input_tokens,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max,)

    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0

    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.bos_token_id
      else:
        x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)

      prev_log_p = None  # lightning: soft masking, the prediction of the previous step of this block

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          # lightning: the same number of masked tokens in every sequence of the batch
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1)[0].item()
          t *= u**(1 / num_masked)
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        soft = None
        if self.soft_masking != 'none' and prev_log_p is not None:  # lightning: binary masks at the first step
          soft = self._sampling_soft_masks(x_accum[:, fwd_idx], prev_log_p)
        p_x0_cache, x_next, log_p = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,
            soft=soft)
        if log_p is not None:
          prev_log_p = log_p
        if p_x0_cache is None:
          sampling_steps += 1

        x_accum[:, fwd_idx] = x_next

      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
    return x_accum, sampling_steps

  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy

  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.

    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x


def owt_datamodule(config, val_tokens):
  """lightning: OpenWebText token streams of data/openwebtext.py (the last 100k documents are the validation split,
  as openwebtext-split in BD3LM; documents separated by <|endoftext|>, wrapped, no BOS / EOS insertion as in
  train_owt_bd3lm.sh), served by data/datamodule.py in chunks of model.length tokens. val_tokens: validation
  budget (evenly spaced chunks), 0 = the whole split."""
  from data.openwebtext import ensure_split

  return TokenDataModule(DataConfig(
    dataset='openwebtext', token_files=ensure_split(100_000, verbose=False), seq_len=config.model.length - 1,
    batch_size=config.loader.batch_size, val_batch_size=config.loader.eval_batch_size, val_tokens=val_tokens,
    num_workers=config.loader.num_workers))
