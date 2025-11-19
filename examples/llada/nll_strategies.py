"""Deterministic unmasking strategies for masked diffusion models allow for *exact* evaluation of negative log-likelihood of a data sample x. 

At each denoising step, a strategy defines 
(1) which *positions* to unmask
(2) how to determine the *value* the unmasked position takes from their vocabulary distribution
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _get_valid_answer_mask(
    batch_size: int,
    seq_len: int, 
    prompt_lens: torch.Tensor,  # [batch_size]
    lengths: torch.Tensor,       # [batch_size]
    device: torch.device
) -> torch.Tensor:
    """
    Get mask indicating valid answer positions for each sample.
    
    Returns:
        is_valid_answer: [batch_size, seq_len] boolean mask
            True for positions in [prompt_len, length)
    """
    # Create position indices [batch_size, seq_len]
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Expand lengths [batch_size, 1] -> [batch_size, seq_len]
    prompt_lens_expanded = prompt_lens.unsqueeze(1)
    lengths_expanded = lengths.unsqueeze(1)

    # Valid answer positions: [prompt_len, length)
    is_valid_answer = (positions >= prompt_lens_expanded) & (positions < lengths_expanded)
        
    return is_valid_answer


# def _greedy_cheating_decoding_nll_step(
#     model: torch.nn.Module, x: torch.Tensor, z_k: torch.Tensor, mask_token_id: int, k: int=1
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     batch_size, seq_len = x.shape
#     is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    
#     logits = model(z_k).logits # [batch_size, seq_len, vocab_size]
#     token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
#     true_log_probs = torch.gather(
#         token_log_probs, dim=-1, index=x.unsqueeze(-1)
#     ).squeeze(-1) # [batch_size, seq_len]
#     masked_log_probs = torch.where(
#         is_masked, true_log_probs, torch.full_like(true_log_probs, float("-inf"))
#     ) # [batch_size, seq_len]
    
#     greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
#     nll_greedy = -masked_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
#     z_k = z_k.clone()  # Avoid in-place modification issues
#     z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]
    
#     return nll_greedy / seq_len, z_k


# def greedy_cheating_decoding_nll(
#     model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
# ) -> torch.Tensor:
#     x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
#     batch_size, seq_len = x.shape
    
#     z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
#     nll_greedy = torch.zeros(batch_size, device=x.device) # [batch_size]
    
#     for num_unmasked in range(seq_len):
#         nll_step, z_k = _greedy_cheating_decoding_nll_step(model, x, z_k, mask_token_id)
#         nll_greedy += nll_step
#     return nll_greedy


def _greedy_decoding_nll_step(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    z_k: torch.Tensor, 
    mask_token_id: int, 
    is_valid_answer: torch.Tensor,
    answer_lens: torch.Tensor,
    k: int=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _ = x.shape
    batch_idx = torch.arange(batch_size, device=x.device)
    
    # Only consider valid answer positions that are still masked
    is_masked = (z_k == mask_token_id) & is_valid_answer # [batch_size, seq_len]
    
    # Early exist if no masked positions remain
    has_masked = is_masked.any(dim=1) # [batch_size]
    if not has_masked.any():
        return torch.zeros(batch_size, device=x.device), z_k

    # Get model predictions
    logits = model(z_k).logits # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    confidence_log_probs = torch.max(token_log_probs, dim=-1).values # [batch_size, seq_len]
    
    # Select position with highest confidence among masked answer positions
    masked_log_probs = torch.where(
        is_masked,
        confidence_log_probs, 
        torch.full_like(confidence_log_probs, float("-inf"))
    ) # [batch_size, seq_len]
    greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    
    # Get true log probs for ground truth tokens
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1) # [batch_size, seq_len]
    
    # Extract NLL at selected positions
    nll_step = -true_log_probs[batch_idx, greedy_position] # [batch_size]
    
    # Zero out NLL for samples that have no masked tokens left
    nll_greedy = torch.where(has_masked, nll_step, torch.zeros_like(nll_step))
    
    # Unmask selected positions
    z_k = z_k.clone()  # Avoid in-place modification issues
    z_k[batch_idx, greedy_position] = x[batch_idx, greedy_position]

    # Normalize by answer length (not total length)
    return nll_greedy / answer_lens, z_k


def greedy_decoding_nll(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    block_size: int, 
    mask_token_id: int, 
    prompt_lens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    # x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    nll_greedy = torch.zeros(batch_size, device=x.device) # [batch_size]

    is_valid_answer = _get_valid_answer_mask(
        batch_size, seq_len, prompt_lens, lengths, x.device
    ) # [batch_size, seq_len]
    mask_fill = torch.full_like(x, mask_token_id) # [batch_size, seq_len]
    z_k = torch.where(is_valid_answer, mask_fill, x)  # [batch_size, seq_len]
    
    answer_lens = lengths - prompt_lens
    max_answer_len = answer_lens.max().item()
    
    for num_unmasked in range(max_answer_len):
        nll_step, z_k = _greedy_decoding_nll_step(
            model, x, z_k, mask_token_id, is_valid_answer, answer_lens
        )
        nll_greedy += nll_step
    return nll_greedy
        

def _greedy_block_decoding_nll_step(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    z_k: torch.Tensor, 
    block_start: torch.Tensor, # [batch_size]
    block_end: torch.Tensor, # [batch_size]
    mask_token_id: int,
    is_valid_answer: torch.Tensor,
    answer_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape
    batch_idx = torch.arange(batch_size, device=x.device)
    
    # Define block region
    seq_idx = torch.arange(seq_len, device=x.device) # [seq_len]
    is_within_block = (seq_idx.unsqueeze(0) >= block_start.unsqueeze(1)) & (seq_idx.unsqueeze(0) < block_end.unsqueeze(1))  # [batch_size, seq_len]
    
    # Only consider valid answer positions that are within block and still masked
    is_masked = (z_k == mask_token_id) & is_valid_answer & is_within_block
    
    # Early exit if no masked positions remain
    has_masked = is_masked.any(dim=1)  # [batch_size]
    if not has_masked.any():
        return torch.zeros(batch_size, device=x.device), z_k
    
    # Get model predictions
    logits = model(z_k).logits  # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    confidence_log_probs = torch.max(token_log_probs, dim=-1).values  # [batch_size, seq_len]
    
    # Select position with highest confidence among masked answer positions in block
    masked_log_probs = torch.where(
        is_masked,
        confidence_log_probs, 
        torch.full_like(confidence_log_probs, float("-inf"))
    )  # [batch_size, seq_len]
    greedy_position = masked_log_probs.argmax(dim=-1)  # [batch_size]
    
    # Get true log probs for ground truth tokens
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len]
    
    # Extract NLL at selected positions
    nll_step = -true_log_probs[batch_idx, greedy_position]  # [batch_size]
    
    # Zero out NLL for samples that have no masked tokens left
    nll_block = torch.where(has_masked, nll_step, torch.zeros_like(nll_step))
    
    # Unmask selected positions
    z_k = z_k.clone()
    z_k[batch_idx, greedy_position] = x[batch_idx, greedy_position]
    
    # Normalize by answer length (not total length)
    return nll_block / answer_lens, z_k


def greedy_block_decoding_nll(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    block_size: int, 
    mask_token_id: int, 
    prompt_lens: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len = x.shape
    nll_greedy = torch.zeros(batch_size, device=x.device)  # [batch_size]
    
    # Get valid answer mask
    is_valid_answer = _get_valid_answer_mask(
        batch_size, seq_len, prompt_lens, lengths, x.device
    )  # [batch_size, seq_len]
    
    # Initialize: mask only answer positions, keep prompt unchanged
    mask_fill = torch.full_like(x, mask_token_id)
    z_k = torch.where(is_valid_answer, mask_fill, x)  # [batch_size, seq_len]
    
    answer_lens = lengths - prompt_lens
    max_answer_len = answer_lens.max().item()
    
    # Define blocks based on answer region, not entire sequence
    num_blocks = math.ceil(max_answer_len / block_size)
    
    for block_idx in range(num_blocks):
        # Answer-relative block boundaries (scalars)
        block_answer_start = block_idx * block_size
        block_answer_end = min(max_answer_len, (block_idx + 1) * block_size)
        
        # Absolute block boundaries (tensors [batch_size] for per-sample prompt handling)
        block_start = prompt_lens + block_answer_start  # [batch_size]
        block_end = prompt_lens + block_answer_end      # [batch_size]
        
        # Unmask positions within current block
        for _ in range(block_answer_start, block_answer_end):
            nll_step, z_k = _greedy_block_decoding_nll_step(
                model, x, z_k, block_start, block_end, mask_token_id, 
                is_valid_answer, answer_lens
            )
            nll_greedy += nll_step
    
    return nll_greedy


def _sample_uniform_mask(
    x: torch.Tensor,                    # [batch_size, seq_len]
    num_unmasked: int,
    mask_token_id: int,
    is_valid_answer: torch.Tensor,      # [batch_size, seq_len]
    answer_lens: torch.Tensor           # [batch_size]
) -> torch.Tensor:
    """
    Fully vectorized version - no per-sample loop.
    
    Strategy:
    1. Assign random priorities to all positions
    2. Sort by priority
    3. Mask top-k positions where k varies per sample
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Start with full sequence
    z_k = x.clone()
    
    # Number of tokens to mask per sample
    num_to_mask = torch.clamp(answer_lens - num_unmasked, min=0)  # [batch_size]
    
    # Assign random priorities to all positions
    random_priorities = torch.rand(batch_size, seq_len, device=device)
    
    # Invalid positions get -inf priority (never selected)
    random_priorities = torch.where(
        is_valid_answer,
        random_priorities,
        torch.full_like(random_priorities, float("-inf"))
    )
    
    # Sort priorities (descending) to find top positions
    sorted_priorities, sorted_indices = torch.sort(
        random_priorities, dim=1, descending=True
    )
    
    # Create mask: True for top-k positions where k = num_to_mask[b]
    position_ranks = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    should_mask = position_ranks < num_to_mask.unsqueeze(1)  # [batch_size, seq_len]
    
    # Map back to original indices using sorted_indices
    # Create scatter target
    mask_pattern = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask_pattern.scatter_(1, sorted_indices, should_mask)
    
    # Apply masking
    z_k = torch.where(mask_pattern, mask_token_id, z_k)
    
    return z_k


def _uniform_decoding_nll_step(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    num_unmasked: int,
    mask_token_id: int,
    is_valid_answer: torch.Tensor,
    prompt_lens: torch.Tensor,
    answer_lens: torch.Tensor,
    num_monte_carlo: int = 32
) -> torch.Tensor:
    batch_size, seq_len = x.shape
    device = x.device
    
    # Check which samples are still active (have tokens left to unmask)
    is_active = (num_unmasked < answer_lens)  # [batch_size]
    
    # Early exit if all samples are done
    if not is_active.any():
        return torch.zeros(batch_size, device=device)
    
    # Number of tokens to mask for each sample
    num_to_mask = torch.clamp(answer_lens - num_unmasked, min=0)  # [batch_size]
    
    nll_accumulator = torch.zeros(batch_size, device=device) # [batch_size]
    for _ in range(num_monte_carlo):
        
        z_k = _sample_uniform_mask(
            x, num_unmasked, mask_token_id, is_valid_answer, answer_lens
        )
        
        # random_permutations = torch.stack(
        #     [torch.randperm(answer_lens[idx], device=x.device) for idx in range(batch_size)]
        # ) # [batch_size, seq_len]
        # positions_to_mask = prompt_lens + random_permutations[:, :num_masked] # [batch_size, num_masked]
        # z_k = x.scatter(dim=-1, index=positions_to_mask, value=mask_token_id)
        
        # Get model predictions
        logits = model(z_k).logits # [batch_size, seq_len, vocab_size]
        token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
        
        # Extract log probs for ground truth tokens
        true_log_probs = torch.gather(
            token_log_probs, dim=-1, index=x.unsqueeze(-1)
        ).squeeze(-1) # [batch_size, seq_len]
        
        # Identify which positions are masked
        is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
        
        # Only consider masked answer positions (not prompt / padding)
        is_masked_answer = is_masked & is_valid_answer # [batch_size, seq_len]
        
        # Sum log probs over masked positions
        masked_log_probs = torch.where(
            is_masked_answer, true_log_probs, torch.zeros_like(true_log_probs)
        ) # [batch_size, seq_len]
        total_log_prob = masked_log_probs.sum(dim=-1) # [batch_size]
        
        # Normalize by number of masked tokens (avoid divide by zero)
        num_to_mask_safe = torch.clamp(num_to_mask, min=1)
        nll_step = -total_log_prob / num_to_mask_safe # [batch_size]

        # Zero out inactive samples and accumulate
        nll_step = torch.where(is_active, nll_step, torch.zeros_like(nll_step)) # [batch_size]
        nll_accumulator += nll_step # [batch_size]

    # Average over Monte Carlo samples
    nll_avg = nll_accumulator / num_monte_carlo # [batch_size]
    
    # Normalize by answer length (not total sequence length)
    answer_len_safe = torch.clamp(answer_lens, min=1)
    nll_normalized = nll_avg / answer_len_safe
    return nll_normalized
        

def uniform_decoding_nll(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    block_size: int, 
    mask_token_id: int, 
    prompt_lens: torch.Tensor, 
    lengths: torch.Tensor
) -> torch.Tensor:
    # x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    
    is_valid_answer = _get_valid_answer_mask(batch_size, seq_len, prompt_lens, lengths, x.device)
    answer_lens = lengths - prompt_lens  # [batch_size]
    max_answer_len = answer_lens.max().item()

    nll_uniform = torch.zeros(batch_size, device=x.device) # [batch_size]
    for num_unmasked in range(max_answer_len):
        nll_step = _uniform_decoding_nll_step(
            model, x, num_unmasked, mask_token_id, is_valid_answer, prompt_lens, answer_lens
        )
        nll_uniform += nll_step
    return nll_uniform


# def _probability_margin_decoding_nll_step(
#     model: torch.nn.Module, x: torch.Tensor, z_k: torch.Tensor, mask_token_id: int, k: int=1
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     batch_size, seq_len = x.shape
#     is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    
#     def _compute_probability_margin(token_log_probs):
#         top_two_log_probs = torch.topk(token_log_probs, k=2, dim=-1).values # [batch_size, seq_len, 2]
#         most_confident = top_two_log_probs[:, :, 0] # [batch_size, seq_len]
#         second_most_confident = top_two_log_probs[:, :, 1] # [batch_size, seq_len]
#         return most_confident - second_most_confident
    
#     logits = model(z_k).logits # [batch_size, seq_len, vocab_size]
#     token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
#     margin_log_probs = _compute_probability_margin(token_log_probs) # [batch_size, seq_len]
#     masked_log_probs = torch.where(
#         is_masked, margin_log_probs, torch.full_like(margin_log_probs, float("-inf"))
#     ) # [batch_size, seq_len]
#     greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    
#     true_log_probs = torch.gather(
#         token_log_probs, dim=-1, index=x.unsqueeze(-1)
#     ).squeeze(-1) # [batch_size, seq_len]
#     nll_greedy = -true_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
#     z_k = z_k.clone()  # Avoid in-place modification issues
#     z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]

#     return nll_greedy / seq_len, z_k


# def probability_margin_decoding_nll(
#     model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
# ) -> torch.Tensor:
#     x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
#     batch_size, seq_len = x.shape
    
#     z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
#     nll_margin = torch.zeros(batch_size, device=x.device) # [batch_size]
    
#     for num_unmasked in range(seq_len):
#         nll_step, z_k = _probability_margin_decoding_nll_step(model, x, z_k, mask_token_id)
#         nll_margin += nll_step
#     return nll_margin


def true_autoregressive_decoding_nll(
    model: torch.nn.Module,
    x: torch.Tensor,                    # [batch_size, seq_len]
    block_size: int,
    prompt_lens: torch.Tensor,          # [batch_size]
    lengths: torch.Tensor               # [batch_size]
) -> torch.Tensor:
    """
    Compute NLL for true autoregressive models (single forward pass).
    
    AR models use causal attention, so one forward pass gives us
    log p(x_i | x_{<i}) for all positions simultaneously.
    
    This ONLY works for models with causal masking (e.g., GPT).
    Does NOT work for masked diffusion models.
    
    Returns:
        nll: [batch_size] total NLL per sample
    """
    # x = x[:, 0:block_size].contiguous()
    batch_size, seq_len = x.shape
    device = x.device
    
    # Get valid answer positions
    is_valid_answer = _get_valid_answer_mask(
        batch_size, seq_len, prompt_lens, lengths, device
    )
    
    answer_lens = lengths - prompt_lens  # [batch_size]
    
    # Single forward pass
    logits = model(x).logits  # [batch_size, seq_len, vocab_size]
    
    # AR models: logits[i] predicts token i+1
    # So we need to shift: compare logits[:-1] with targets[1:]
    logits_shifted = logits[:, :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
    targets_shifted = x[:, 1:].contiguous()  # [batch_size, seq_len-1]
    
    # Also shift the validity mask
    is_valid_answer_shifted = is_valid_answer[:, 1:]  # [batch_size, seq_len-1]
    
    # Compute log probs
    token_log_probs = F.log_softmax(logits_shifted, dim=-1)
    
    # Extract log probs for ground truth tokens
    true_log_probs = torch.gather(
        token_log_probs,
        dim=2,
        index=targets_shifted.unsqueeze(2)
    ).squeeze(2)  # [batch_size, seq_len-1]
    
    # Only sum over valid answer positions
    masked_log_probs = torch.where(
        is_valid_answer_shifted,
        true_log_probs,
        torch.zeros_like(true_log_probs)
    )
    
    # Sum and normalize
    total_log_prob = masked_log_probs.sum(dim=1)  # [batch_size]
    nll = -total_log_prob  # [batch_size]
    
    # Normalize by answer length
    answer_lens_safe = torch.clamp(answer_lens, min=1)
    nll_normalized = nll / answer_lens_safe
    
    return nll_normalized
    


def compute_deterministic_nll(
    x: torch.Tensor, 
    model: torch.nn.Module, 
    decoding_strategy: str,
    mask_token_id: int,
    block_size: int,
    prompt_lens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if decoding_strategy == "greedy":
        nll = greedy_decoding_nll(model, x, block_size, mask_token_id, prompt_lens, lengths)
    elif decoding_strategy == "block-greedy":
        nll = greedy_block_decoding_nll(model, x, block_size, mask_token_id, prompt_lens, lengths)
    elif decoding_strategy == "uniform":
        nll = uniform_decoding_nll(model, x, block_size, mask_token_id, prompt_lens, lengths)
    # elif decoding_strategy == "probability-margin":
    #     nll = probability_margin_decoding_nll(model, x, block_size, mask_token_id, prompt_lens, lengths)
    # elif decoding_strategy == "greedy-cheating":
    #     nll = greedy_cheating_decoding_nll(model, x, block_size, mask_token_id, prompt_lens, lengths)
    elif decoding_strategy == "auto-regressive":
        nll = true_autoregressive_decoding_nll(model, x, block_size, prompt_lens, lengths)
    else:
        raise ValueError(f"Unknown unmasking strategy: {decoding_strategy}")
    return nll