"""
accelerate launch \
    --num_processes 2 \
    dllm/pipelines/llada/eval.py \
    --tasks gsm8k \
    --model llada \
    --num_fewshot 8 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"
"""

import math
from types import SimpleNamespace
from dataclasses import dataclass

import accelerate
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

import dllm
from dllm.pipelines.llada import LLaDAGenerator, LLaDAGeneratorConfig

@dataclass
class LLaDAEvalConfig(LLaDAGeneratorConfig):
    max_new_tokens: int = 1024
    max_length: int = 4096
    steps: int = 1024
    block_length: int = 1024
    
    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = True
    device: str = "cuda"


@register_model("llada")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None: config = LLaDAEvalConfig()

        # Pull args from config, allow kwargs to override
        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_length = kwargs.get("block_length", config.block_length)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        
        accelerator = accelerate.Accelerator()

        # Get GLOBAL rank from torch.distributed (not accelerator)
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()        # ← GLOBAL rank (0-15)
            self._world_size = torch.distributed.get_world_size()  # ← GLOBAL world size (16)
        else:
            self._rank = 0
            self._world_size = 1

        # Use accelerator for device placement
        self.model = dllm.utils.get_model(SimpleNamespace(
            model_name_or_path=pretrained,
            dtype=get_dtype(dtype)
        ))
        self.model.eval()

        if accelerator.num_processes > 1:
            # Let accelerator handle device placement
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device  # ← Accelerator figures out local device correctly
            self.accelerator = accelerator
        else:
            # Single GPU
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = dllm.utils.get_tokenizer(SimpleNamespace(
            model_name_or_path=pretrained, 
            model=self.model
            ))
    
        # generation params
        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.max_new_tokens = int(max_new_tokens)
        self.block_length = int(block_length)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy

        # loglikelihood params
        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.   

    def apply_chat_template(self, chat_history: list[dict[str, str]],add_generation_prompt: bool = True) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch: torch.Tensor, prompt_index: torch.Tensor) -> torch.Tensor:
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """Compute ELBO, a bound on the true loglikelihood"""
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none") / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)
        
    def _exact_loglikelihood_step(
        self, 
        seq: torch.Tensor, 
        prompt_index: torch.Tensor, 
        z_k: torch.Tensor, 
        block_start: int, 
        block_end: int
    ):
        """Assume batch_size=1"""
        batch_size, seq_len = seq.shape
        target_len = seq_len - prompt_index.sum()
        
        logits = self.model(z_k).logits # [batch_size, seq_len, vocab_size]
        token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
        confidence_log_probs = torch.max(token_log_probs, dim=-1).values # [batch_size, seq_len]
        
        is_mask_token = (z_k == self.mask_id) # [batch_size, seq_len]
        seq_idx = torch.arange(seq_len, device=seq.device) # [seq_len]
        is_within_block = (block_start <= seq_idx) & (seq_idx < block_end) # ensures prompt is never evaluated
        bool_mask = is_mask_token & is_within_block
        
        masked_log_probs = torch.where(
            bool_mask, confidence_log_probs, torch.full_like(confidence_log_probs, float("-inf"))
        ) # [batch_size, seq_len]
        greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
        
        true_log_probs = torch.gather(
            token_log_probs, dim=-1, index=seq.unsqueeze(-1)
        ).squeeze(-1) # [batch_size, seq_len]
        nll_greedy = -true_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
        z_k = z_k.clone()  # Avoid in-place modification issues
        z_k[torch.arange(batch_size), greedy_position] = seq[torch.arange(batch_size), greedy_position]
        
        return nll_greedy / target_len, z_k
    
    @torch.no_grad()
    def get_exact_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """Compute the exact loglikelihood under greedy block-decoding (assumes batch_size=1)"""
        seq = torch.concatenate([prefix, target])[None, :].to(self.device) # [1, seq_len]
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        
        z_k = torch.full_like(seq, self.mask_id, device=self.device) # [1, seq_len]
        z_k[:, :len(prefix)] = seq[:, :len(prefix)]  # Keep prefix unmasked
        nll_block_greedy = torch.tensor([0.0], device=self.device) # []
        
        num_blocks = math.ceil(len(target) / self.block_length)
        for block_idx in range(num_blocks):
            block_start = len(prefix) + self.block_length * block_idx
            block_end = len(prefix) + min(self.block_length * (block_idx + 1), len(target))
            
            for _ in range(block_start, block_end):
                nll_step, z_k = self._exact_loglikelihood_step(
                    seq, prompt_index, z_k, block_start, block_end
                )
                nll_block_greedy += nll_step
        return -nll_block_greedy.item()

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix: torch.Tensor, target: torch.Tensor) -> bool:
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context: str, continuation: str) -> tuple[torch.Tensor, torch.Tensor]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                # ll = self.get_exact_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]["until"]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        generator = LLaDAGenerator(model=self.model, tokenizer=self.tokenizer)
        
        for elem in tqdm(ds, desc="Generating..."):
            prompt = [elem["question"].to(self.device)]
            stop_tokens = elem["until"]
            generated_ids = generator.generate(
                inputs=prompt, 
                steps=self.steps, 
                max_new_tokens=self.max_new_tokens, 
                block_length=self.block_length, 
                temperature=0.0, 
                cfg_scale=self.cfg, 
                remasking=self.remasking)
            generated_answer = self.tokenizer.decode(generated_ids[0][prompt[0].shape[0]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    cli_evaluate()
