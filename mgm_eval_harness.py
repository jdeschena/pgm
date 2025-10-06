import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import algo
import dataloader
import os


"""
# Instructions on running the eval

## What is the script doing?
The script evaluates (or approximates) the log-likelihood of
prefix + suffix. The script will load a checkpoint, and 
depending on the config, use the corresponding class.

- For MCQ, the log-likelihood of all continuations given the 
  same prefix is evaluated. The most likely continuation is 
  selected as the "correct" answer according to the model.

- For lambada_openai, the model is correct if the true 
  continuation is generated as the argmax. For diffusion, we 
  inject noise in the continuation, and check whether the true 
  answer computed in a single forward pass is the most likely. 
  This is naturally favoring AR models, as they run one forward 
  pass per token, while diffusion use a single pass for all 
  tokens for simplicity. Therefore, I usually only compare on 
  MCQ since it is more fair.

To run the script, you need to install the lm-eval-harness package:
```
pip install git+https://github.com/EleutherAI/lm-evaluation-harness
```

## Tasks that we tested with
**MCQ**: arc_easy, arc_challenge, hellaswag, winogrande, 
         boolq, openbookqa, race, social_iqa, mathqa, piqa
**Likelihood based**: lambada_openai

## Batch size that fits at the small scale
    - boolq -> 64
    - openbookqa -> 256
    - race -> 32
    - social_iqa -> 512
    - winogrande -> 64
    - mathqa -> 64
    - lambada_openai -> 64
    - arc_easy -> 256
    - arc_challenge -> 256
    - hellaswag -> 64
    - piqa -> 64

## Important flags
  --trust_remote_code -> some datasets execute code when loading 
                         from huggingface. Without this flag, 
                         the script might crash.
  --batch_size        -> max. num elements to use in parallel 
                         to eval the likelihood (for diffusion). 
                         For simplicity, inputs are NOT padded 
                         and batched for AR, though it should 
                         be fairly easy to add.
  --tasks             -> one or multiple comma-separated tasks 
                         to evaluate on.
  --model_args        -> string (without spaces) that contains 
                         the arguments to pass to the evaluator 
                         (stuff like checkpoints path, number 
                         of MC samples to evaluate the 
                         likelihood, path to the sentencepiece 
                         tokenizer, etc).
  --output_path       -> path to a json file where the evaluation 
                         results will be saved, instead of 
                         only being printed in the terminal 
                         (they will always be printed).
  --limit             -> debugging flag; limit the number of 
                         examples to a fixed amount, instead 
                         of using the whole dataset


## Example commands

### Run a single task with an MDLM model, and 2048 MC samples to approximate the likelihood (we used 1024 in the paper)
python diffusion_harness.py \
    --tasks arc_easy \
    --model mgm \
    --batch_size 256 \
    --model_args checkpoint_path=/home/username/baselines/mdlm/1M.ckpt,num_mc_samples=2048 \
    --output_path ./harness_results/mdlm/1M/arc_easy.ckpt

### Debug with 20 examples only
python diffusion_harness.py \
    --tasks arc_easy \
    --model mgm \
    --batch_size 256 \
    --model_args checkpoint_path=/home/username/baselines/mdlm/1M.ckpt,num_mc_samples=2048 \
    --limit 20 \
    --output_path ./harness_results/mdlm/1M/arc_easy.ckpt


"""


def requests_to_dataset(config, requests, tokenizer, num_proc):
  def _tokenize(e):
    eos_idx = tokenizer.eos_token_id
    bos_idx = tokenizer.bos_token_id
    prefix_tokens = tokenizer(e['prefix'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    target_tokens = tokenizer(e['target'], 
                              return_attention_mask=False, 
                              add_special_tokens=False
                              )['input_ids']
    prefix_tokens = [bos_idx] + prefix_tokens
    target_tokens = target_tokens + [eos_idx]
    
    return {
        'prefix_text': e['prefix'],
        'target_text': e['target'],
        'prefix': prefix_tokens,
        'target': target_tokens,
    }
  ds = []
  ds = [{'prefix': req.args[0], 'target': req.args[1]} 
        for req in requests]
  ds = Dataset.from_list(ds)
  ds = ds.map(_tokenize, num_proc=num_proc)
  ds = ds.with_format('torch')
  seq_lenths = [len(x['prefix']) + len(x['target']) 
                for x in ds]
  
  num_larger = len([x for x in seq_lenths if x > config.model.length])
  if num_larger > 0:
    print(f'\033[91mThere are some examples that are longer '
          f'than the context length, they will be ignored '
          f'during evaluation. Number of such sequences: '
          f'{num_larger}\033[0m')

  return ds


def _eval_suffix_nll_generators(config, module, prefix, suffix,
                                batch_size, num_samples):
  device = module.device
  assert num_samples % batch_size == 0
  full_sentence = torch.cat([prefix, suffix], dim=-1
                  ).repeat(batch_size, 1).to(module.device)
  all_ts = module._sample_t(num_samples, accum_step=None)
  for idx in range(0, num_samples, batch_size):
    t = all_ts[idx:idx+batch_size].unsqueeze(-1)
    dalpha_t, alpha_t = module.noise(t)
    alpha_t = alpha_t.to(device)
    sigma = module._sigma_from_alphat(alpha_t)
    x0 = full_sentence.to(device)
    if config.algo.name == 'mdlm':
      xt = module.q_xt(full_sentence, alpha_t).to(device)
      group_idxs = None
    elif config.algo.name == 'partition-mdlm':
      xt = x0.to(device)
      group_idxs = module._q_xt_partition(x0, alpha_t)
    else:
      raise ValueError(config.algo.name)
    yield xt, x0, group_idxs, sigma, alpha_t, dalpha_t
    

def eval_suffix_nll(config, module, prefix, suffix, batch_size, 
                    num_samples):
  all_losses = []
  generator =  _eval_suffix_nll_generators(config, module, 
                  prefix, suffix, batch_size, num_samples)
  for xt, x0, group_idxs, sigma, alpha_t, dalpha_t in generator:
    cond = torch.zeros_like(sigma)  # No time conditioning
    if group_idxs is not None:
      log_x_theta = module(xt, cond, group_idxs=group_idxs)
    else:
      log_x_theta = module(xt, cond)
    token_nll = module.nll_per_token(log_x_theta, xt, x0, 
                                     alpha_t, dalpha_t)
    if group_idxs is not None:
      # Assume group 1 is masked
      token_nll = token_nll * group_idxs
    all_losses.append(float(token_nll.mean()))
  return float(np.mean(all_losses))


@register_model("mgm")
class MGMEvalWrapper(LM):
  def __init__(self, pretrained="NONE", max_length=1024,
               num_mc_samples=1024, batch_size=64, device="cuda",
               checkpoint_path=None, num_proc=8, *args, **kwargs):
    super().__init__()
    if not os.path.exists(checkpoint_path):
      raise ValueError(f'{checkpoint_path=} doesn\' exist.')
    ckpt = torch.load(checkpoint_path, map_location='cpu', 
                      weights_only=False)
    config = ckpt['hyper_parameters']['config']

    self.tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'mdlm':
      self.model = algo.MDLM(config, self.tokenizer)
    elif config.algo.name == 'partition-mdlm':
      self.model = algo.PartitionMDLM(config, 
                                      self.tokenizer)
    else:
      raise ValueError(f'Implement for {config.algo.name}')
    self.config = config
    self.num_proc = num_proc
    self.num_mc_samples = num_mc_samples
    self.batch_size = int(batch_size)
    self.device = device

    self.model.load_state_dict(ckpt['state_dict'])
    self.model.to(device)
    self.model.eval()

  def suffix_greedy_prediction(self, prefix, target):
    if self.config.algo.name == 'mdlm':
      return self._suffix_greedy_prediction_mdlm(prefix, 
                                                 target)
    elif self.config.algo.name == 'partition-mdlm':
      return self._suffix_greedy_prediction_pgm(prefix, 
                                                target)
    else:
      raise ValueError(self.config.algo.name)

  def _suffix_greedy_prediction_mdlm(self, prefix, target):
    mask_idx = self.model.mask_index
    eos_idx = self.tokenizer.eos_token_id
    noisy_target = [mask_idx] * (len(target) - 1) + [eos_idx]
    noisy_target = torch.tensor(noisy_target, 
                                device=self.device)
    prefix = prefix.to(self.device)
    seq = torch.concatenate([prefix, noisy_target], 
                            dim=-1).reshape(1, -1)
    sigma = torch.zeros(size=(seq.shape[0], 1), 
                        device=self.device)
    logits = self.model(seq, sigma)
    assert logits.shape[0] == 1
    suffix_logits = logits[0, len(prefix):]
    target_preds = suffix_logits.argmax(-1).cpu()
    correct = target_preds == target
    correct = correct.all()
    return bool(correct)
  
  def _suffix_greedy_prediction_pgm(self, prefix, target):
    eos_idx = self.tokenizer.eos_token_id
    seq = torch.concatenate([prefix, target], dim=-1)[None]
    seq[0, -1] = eos_idx
    group_idxs = torch.tensor([0] * len(prefix) 
                            + [1] * (len(target) - 1) 
                            + [0], device=self.device)[None]
    sigma = torch.zeros(size=(seq.shape[0], 1),
                        device=self.device)
    logits = self.model(seq, sigma, group_idxs=group_idxs)
    assert logits.shape[0] == 1
    suffix_logits = logits[0, len(prefix):-1]
    target_preds = suffix_logits.argmax(-1).cpu()
    correct = target_preds == target[:-1]
    correct = correct.all()
    return bool(correct)
    
  @torch.no_grad()
  def loglikelihood(self, requests: list[Instance]) \
                                -> list[tuple[float, bool]]:
    dataset = requests_to_dataset(self.config, requests, 
                                  self.tokenizer, self.num_proc)
    model_length = self.model.config.model.length
    out = []
    for elem in tqdm(dataset, 'Computing likelihood...'):
      prefix = elem['prefix']
      target = elem['target']
      
      if len(prefix) + len(target) > model_length:
        ll = 0.0
        is_target_greedy_dec = False
        out.append((ll, is_target_greedy_dec))
        continue

      ll = -eval_suffix_nll(self.config, self.model, prefix, 
                            target, self.batch_size, 
                            self.num_mc_samples)
      is_target_greedy_dec = self.suffix_greedy_prediction(
        prefix, target)
      out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
    return out

  def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
    raise NotImplementedError
  
  def generate_until(self, context, max_length, stop, 
                     **generation_kwargs):
    raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()