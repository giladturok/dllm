"""
evaluate_nll.py - Simple NLL evaluation for masked diffusion models

Usage:
    # Single GPU (simple)
    python evaluate_nll.py --model_path GSAI-ML/LLaDA-8B-Base --strategy greedy
    
    # With accelerate (if you want to scale later)
    accelerate launch evaluate_nll.py --model_path GSAI-ML/LLaDA-8B-Base --strategy greedy
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from dllm.utils.model_utils import get_model, get_tokenizer
from dllm.utils.configs import ModelArguments
from nll_strategies import compute_deterministic_nll


DATASET_FORMATS = {
    "wikitext": {
        "type": "text",  # Unconditional - compute NLL on entire text
        "text_field": "text",
    },
    "gsm8k": {
        "type": "conditional",  # Conditional - prompt + target
        "prompt_field": "question",
        "target_field": "answer"
    },
}


def _get_dataset_format(dataset_name):
    """Get format config, default to text format"""
    return DATASET_FORMATS.get(
        dataset_name,
        {"dataset_config": dataset_name, "type": "text", "text_field": "text"}  # Default
    )


@dataclass
class EvalArgs:
    """Simple evaluation arguments"""
    model_path: str
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    strategy: str = "greedy"
    block_size: int = 32
    batch_size: int = 32
    max_samples: int = -1
    max_seq_len: int = 4096
    output_dir: str = "eval_results"


class SimpleNLLEvaluator:
    """Simple single-GPU NLL evaluator"""
    
    def __init__(self, args: EvalArgs):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {args.model_path}")
        
        # Use your existing utilities
        model_args = ModelArguments(
            model_name_or_path=args.model_path,
            dtype="bfloat16"
        )
        self.tokenizer = get_tokenizer(model_args)
        self.model = get_model(model_args)
        self.model.to(self.device).eval()
        
        # Get mask token
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is not None:
            print(f"Mask token ID: {self.mask_token_id}")
        else:
            print("No mask token found.")
        
        print(f"Strategy: {args.strategy}")
        print(f"Device: {self.device}")
    
    def load_dataset(self):
        """Load and tokenize dataset"""
        print(f"\nLoading dataset: {self.args.dataset}")
        
        # Load dataset
        if self.args.dataset_config:
            dataset = load_dataset(
                self.args.dataset,
                self.args.dataset_config,
                split=self.args.split
            )
        else:
            dataset = load_dataset(self.args.dataset, split=self.args.split)
        
        # Limit samples
        if self.args.max_samples > 0:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))
        
        print(f"Total samples: {len(dataset)}")
        
        # Tokenize
        def tokenize_fn(examples):
            fmt = _get_dataset_format(self.args.dataset)
            
            if fmt["type"] == "text":
                # Unconditional: compute NLL on entire text
                if fmt["text_field"] not in examples:
                    raise ValueError(
                        f"Text field '{fmt['text_field']}' not found. "
                        f"Available: {list(examples.keys())}"
                    )
                
                tokenized = self.tokenizer(
                    examples[fmt["text_field"]],
                    truncation=True,
                    max_length=self.args.max_seq_len,
                    padding=False
                )
                # Mark that entire sequence is target
                tokenized["prompt_len"] = [0] * len(tokenized["input_ids"])
            
            elif fmt["type"] == "conditional":
                # Conditional: prompt + target
                prompt_field = fmt["prompt_field"]
                target_field = fmt["target_field"]
                
                prompts = examples[prompt_field]
                targets = examples[target_field]
                formatted_prompts = [f"Question: {q}\nAnswer: " for q in prompts]
                
                # Tokenize separately to track lengths
                prompt_tokens = self.tokenizer(formatted_prompts, add_special_tokens=True)
                target_tokens = self.tokenizer(targets, add_special_tokens=False)
                
                # Concatenate
                input_ids, prompt_lens = [], []
                for p_ids, t_ids in zip(prompt_tokens["input_ids"], target_tokens["input_ids"]):
                    full_ids = p_ids + t_ids
                    # Truncate if needed
                    if len(full_ids) > self.args.max_seq_len:
                        # Keep full prompt, truncate target
                        if len(p_ids) >= self.args.max_seq_len:
                            full_ids = p_ids[:self.args.max_seq_len]
                            prompt_len = self.args.max_seq_len
                        else:
                            target_budget = self.args.max_seq_len - len(p_ids)
                            full_ids = p_ids + t_ids[:target_budget]
                            prompt_len = len(p_ids)
                    else:
                        prompt_len = len(p_ids)
                    
                    input_ids.append(full_ids)
                    prompt_lens.append(prompt_len)
                
                tokenized = {
                    "input_ids": input_ids,
                    "prompt_len": prompt_lens
                }
            
            else:
                raise ValueError(f"Unknown dataset type: {fmt['type']}")
            
            return tokenized
        
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=10,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) >= 10,
            desc="Filtering short sequences"
        )
        
        dataset.set_format("torch")
        return dataset
    
    def collate_fn(self, batch):
        """Pad batch and track prompt lengths"""
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids = []
        lengths = []
        prompt_lens = []
        
        for item in batch:
            ids = item["input_ids"]
            length = len(ids)
            prompt_len = item.get("prompt_len", 0)  # 0 means compute on full seq
            
            lengths.append(length)
            prompt_lens.append(prompt_len)
            
            # Pad
            if length < max_len:
                padding = torch.full(
                    (max_len - length,),
                    self.tokenizer.pad_token_id,
                    dtype=ids.dtype
                )
                ids = torch.cat([ids, padding])
            
            input_ids.append(ids)
        
        return {
            "input_ids": torch.stack(input_ids),
            "lengths": torch.tensor(lengths),
            "prompt_lens": torch.tensor(prompt_lens)
        }
    
    @torch.no_grad()
    def evaluate(self):
        """Main evaluation loop"""
        dataset = self.load_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        all_nlls = []
        all_nlls_per_token = []
        all_lengths = []
        
        # Running statistics for progress bar
        running_sum = 0.0
        running_count = 0
        
        print(f"\nEvaluating with strategy: {self.args.strategy}")
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            prompt_lens = batch["prompt_lens"].to(self.device)
            
            # Compute NLL
            nll = compute_deterministic_nll(
                x=input_ids,
                model=self.model,
                decoding_strategy=self.args.strategy,
                mask_token_id=self.mask_token_id,
                block_size=self.args.block_size,
                prompt_lens=prompt_lens,
                lengths=lengths
            )  # Returns per-token NLL [batch_size]
            
            # IMPORTANT: Adjust for prompt-only datasets
            # NLL is computed over full sequence, but we only want target portion
            target_lens = lengths - prompt_lens  # Length of answer portion
            
            # Store results (only count target tokens)
            nll_cpu = nll.cpu()
            target_lens_cpu = target_lens.cpu()
            
            all_nlls_per_token.extend(nll_cpu.tolist())
            all_nlls.extend((nll_cpu * target_lens_cpu).tolist())  # Total NLL on targets
            all_lengths.extend(target_lens_cpu.tolist())  # Target lengths only
            
            # Update running statistics
            running_sum += nll_cpu.sum().item()
            running_count += len(nll_cpu)
            running_avg = running_sum / running_count
            
            # Update progress bar with running metrics
            progress_bar.set_postfix({
                'avg_nll': f'{running_avg:.4f}',
                'ppl': f'{np.exp(running_avg):.2f}',
                'samples': running_count
            })
        
        self.save_results(all_nlls, all_nlls_per_token, all_lengths)
    
    def save_results(self, all_nlls, all_nlls_per_token, all_lengths):
        """Save results to JSON"""
        all_nlls = np.array(all_nlls)
        all_nlls_per_token = np.array(all_nlls_per_token)
        all_lengths = np.array(all_lengths)
        
        results = {
            "config": {
                "model": self.args.model_path,
                "dataset": self.args.dataset,
                "split": self.args.split,
                "strategy": self.args.strategy,
                "num_samples": len(all_nlls)
            },
            "metrics": {
                # Per-token (for fair comparison)
                "nll_per_token": {
                    "mean": float(np.mean(all_nlls_per_token)),
                    "std": float(np.std(all_nlls_per_token)),
                    "median": float(np.median(all_nlls_per_token)),
                    "min": float(np.min(all_nlls_per_token)),
                    "max": float(np.max(all_nlls_per_token))
                },
                # Aggregate
                "total_nll": float(np.sum(all_nlls)),
                "total_tokens": int(np.sum(all_lengths)),
                "mean_length": float(np.mean(all_lengths)),
                "perplexity": float(np.exp(np.mean(all_nlls_per_token)))
            }
        }
        
        # Extract model name from path
        model_name = Path(self.args.model_path).name  # e.g., "llama-3-8b" or checkpoint name
        
        # Build hierarchical directory structure
        output_dir = Path(self.args.output_dir) / self.args.dataset / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{self.args.split}_{self.args.strategy}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Results: {self.args.strategy}")
        print(f"{'='*60}")
        print(f"Samples:      {len(all_nlls)}")
        print(f"Total tokens: {int(np.sum(all_lengths))}")
        print(f"\nPer-Token Metrics:")
        print(f"  Mean NLL:   {results['metrics']['nll_per_token']['mean']:.4f}")
        print(f"  Std:        {results['metrics']['nll_per_token']['std']:.4f}")
        print(f"  Perplexity: {results['metrics']['perplexity']:.2f}")
        print(f"\nSaved to: {output_file}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLL for MDLMs")
    
    # Required
    parser.add_argument("--model_path", type=str, required=True)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    
    # Evaluation
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "block-greedy", "uniform", "probability-margin", "greedy-cheating", "auto-regressive"]
    )
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=-1)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results")
    
    args = parser.parse_args()
    
    # Convert to dataclass
    eval_args = EvalArgs(
        model_path=args.model_path,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        strategy=args.strategy,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator = SimpleNLLEvaluator(eval_args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()