import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, List, Dict, Any
from pathlib import Path
import glob, re, os
import queue
import threading

import json
import time
import hashlib
import fcntl

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import math
from collections import defaultdict, deque

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl import DataProto
import numpy as np
from typing import Dict, Any, List
from verl.utils.debug import marked_timer

def _rand_like_compat(t: torch.Tensor, seed: int | None = None):
    if seed is None:
        return torch.rand_like(t)
    g = torch.Generator(device=t.device if t.device.type == "cpu" else "cpu")
    g.manual_seed(seed)
    try:
        return torch.rand(t.shape, device=t.device, dtype=t.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.rand_like(t)

@torch.no_grad()
def spec_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
):

    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    # Δlogp = new - old
    log_ratio = (new_logp - old_logp).masked_fill(~valid, 0.0)

    U = _rand_like_compat(new_logp, seed=seed).clamp_min(1e-12)
    logU = torch.log(U)

    # accept condition: (Δ>=0) or (logU <= Δ)
    accept_mask = (~valid) | (log_ratio >= 0) | (logU <= log_ratio)

    # optional absolute threshold: require new_logp >= log(p_abs_thresh)
    if p_abs_thresh is not None:
        import math
        log_th = math.log(p_abs_thresh)
        accept_mask &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept_mask)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)  # no bad = 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = (resp_len - cut_idx).clamp(min=0)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


@torch.no_grad()
def spec_cut_with_knobs(
    old_logp: torch.Tensor,         # [B,R]
    new_logp: torch.Tensor,         # [B,R]
    response_mask: torch.Tensor,    # [B,R] 1=valid response token
    *,
    bias: float = 0.0,              # b:  >0 more strict; <0 more lenient
    scale: float = 1.0,             # s:  <1 more lenient; >1 more strict
    p_abs_thresh: float | None = None,  # optional: new >= 0.3
    seed: int | None = None,
):
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim()==2
    B, R   = new_logp.shape
    valid  = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    delta  = (new_logp - old_logp).masked_fill(~valid, 0.0)  # Δ

    delta2 = scale * (delta - bias)

    logU = torch.log(_rand_like_compat(new_logp, seed=seed).clamp_min(1e-12))
    accept = (~valid) | (delta2 >= 0) | (logU <= delta2)
    # import pdb; pdb.set_trace()

    if p_abs_thresh is not None:
        log_th = math.log(p_abs_thresh)
        accept &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)       # Return 0 when there are no bad tokens -> 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)     # [B]

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)

    # note: saved tokens ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/bias": float(bias), "spec/scale": float(scale),

    }
    return {
        "cut_idx": cut_idx,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "resp_len": resp_len,
        "per_request_max_new_tokens": (R - cut_idx).to(torch.long),
        "metrics": metrics
        
    }


@torch.no_grad()
def rand_reuse_cut(
    old_logp: torch.Tensor,          # [B,R] 
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # ∈[0,1]
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select rows with probability reuse_prob ----
    # only need per-row random number: reuse -> cut_idx=resp_len; otherwise -> cut_idx=0
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    reuse_mask = (row_rand < reuse_prob)                                    # [B] bool
    need_mask  = ~reuse_mask

    # ---- calculate cut_idx / idx_* / per_request_max_new_tokens----
    zero_like = torch.zeros_like(resp_len)
    cut_idx   = torch.where(reuse_mask, resp_len, zero_like)                # [B]

    # ---- additional check: ensure cut_idx is in valid range ----
    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)       # [Nr]
    idx_need  = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)       # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)               # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    # "saved tokens" use the same definition as spec_cut_with_knobs: saved ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_p":   float(reuse_prob),
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }

@torch.no_grad()
def rand_reuse_all_cut(
    old_logp: torch.Tensor,          # [B,R]
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # probability to randomly select truncation for all data
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select truncation point with probability reuse_prob ----
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    cut_idx   = torch.floor(row_rand * resp_len.to(torch.float32)).to(torch.long)  # [B]
    
    # ensure cut_idx is in valid range

    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    # ---- calculate idx_* / per_request_max_new_tokens (align with spec_*)----
    reuse_mask = (cut_idx == resp_len)                       # [B] bool
    need_mask  = ~reuse_mask                                 # [B] bool
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)  # [Nr]
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)  # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)  # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_all_p":   float(reuse_prob),  # new: random truncation probability
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }



def build_ctx(p_ids, p_msk, p_pos,
                      response_ids,          # [N, R]
                      cut_idx,               # [N]
                      pad_id: int):
    # convention: p_ids/msk/pos shape are [N, P] (sub-batch of idx_need from gen_batch)
    N, P = p_ids.shape
    R    = response_ids.shape[1]
    k_vec = cut_idx.clamp(min=0, max=R)          # [N]
    max_k = int(k_vec.max().item()) if N > 0 else 0
    ctx_len = P + max_k

    # calculate actual length of each prompt (right-aligned non-pad segment)
    Lp = p_msk.sum(dim=1)                         # [N]
    start = ctx_len - (Lp + k_vec)                # [N] length of left pad

    # prepare column index [ctx_len]
    col = torch.arange(ctx_len, device=p_ids.device).unsqueeze(0)   # [1, ctx_len]
    start_ = start.unsqueeze(1)                                     # [N,1]
    Lp_    = Lp.unsqueeze(1)
    k_     = k_vec.unsqueeze(1)

    # 1) boolean mask and source column for prompt segment
    prom_mask = (col >= start_) & (col < start_ + Lp_)              # [N, ctx_len]
    # map columns of ctx back to columns of p_ids: right-aligned with Lp
    src_prom_col = (P - Lp_ ) + (col - start_)                      # [N, ctx_len]
    src_prom_col = src_prom_col.clamp(0, P-1)

    # 2) boolean mask and source column for prefix segment
    pref_mask = (col >= start_ + Lp_) & (col < start_ + Lp_ + k_)   # [N, ctx_len]
    # first cut old responses of each row to max_k, for gather
    resp_cut = response_ids[:, :max_k]                              # [N, max_k]
    src_pref_col = (col - (start_ + Lp_)).clamp(0, max_k-1)         # [N, ctx_len]

    # 3) assemble ctx_ids (default full pad, then fill two segments with where)
    ctx_ids = torch.full((N, ctx_len), pad_id, dtype=p_ids.dtype, device=p_ids.device)
    # prompt segment from the tail of p_ids
    # import pdb; pdb.set_trace()

    prom_vals = torch.gather(p_ids, 1, src_prom_col)                # [N, ctx_len]
    ctx_ids = torch.where(prom_mask, prom_vals, ctx_ids)
    # prefix segment from response prefix

    pref_vals = torch.gather(resp_cut, 1, src_pref_col)             # [N, ctx_len]
    ctx_ids = torch.where(pref_mask, pref_vals, ctx_ids)

    # 4) attention_mask
    ctx_msk = (prom_mask | pref_mask).to(p_msk.dtype)               # [N, ctx_len]

    # 5) position_ids (prompt segment copy original position, prefix segment increment)
    ctx_pos = torch.zeros((N, ctx_len), dtype=p_pos.dtype, device=p_pos.device)
    prom_pos_vals = torch.gather(p_pos, 1, src_prom_col)            # [N, ctx_len]
    ctx_pos = torch.where(prom_mask, prom_pos_vals, ctx_pos)

    last_pos = p_pos[:, -1].unsqueeze(1)                            # [N,1] original right end position
    ar = torch.arange(1, max_k+1, device=p_pos.device).unsqueeze(0) # [1, max_k]
    pref_pos_table = last_pos + ar                                  # [N, max_k]
    pref_pos_vals = torch.gather(pref_pos_table, 1, src_pref_col.clamp(0, max_k-1))
    ctx_pos = torch.where(pref_mask, pref_pos_vals, ctx_pos)

    return ctx_ids, ctx_msk, ctx_pos, max_k


def align_prev_to_gen(
    *,
    cached_tensors: Dict[int, Dict[str, torch.Tensor]],
    gen_batch: Any,
    pad_token_id: int = 151643, 
) -> Dict[str, torch.Tensor]:
    """
    将哈希缓存中的数据，严格按 batch 行号对齐，并补全未命中的行为 dummy。
    核心原则：未命中的行必须保留真实的 Prompt 信息，仅将历史复用信息置空。
    """
    batch_size = len(gen_batch.batch["input_ids"])

    # 1. 获取一个样本的 shape，用于创建 dummy tensor 的形状
    first_hit_idx = list(cached_tensors.keys())[0]
    sample = cached_tensors[first_hit_idx]
    
    dummy_logp = torch.zeros_like(sample["old_logps"])
    dummy_resp = torch.full_like(sample["response_ids"], pad_token_id)

    log_probs_list = []
    response_masks_list = []
    prompts_list = []
    responses_list = []
    position_ids_list = []

    # 2. 按 batch 绝对行号遍历 (0, 1, 2 ... batch_size-1)
    for idx in range(batch_size):
        if idx in cached_tensors:
            # ---- 命中：直接提取并计算 mask/pos ----
            data = cached_tensors[idx]
            resp_ids = data["response_ids"]
            in_ids = data["input_ids"]
            old_logps = data["old_logps"]

            
        else:
            # ---- 未命中：保留真实 Prompt，历史复用信息置空 ----
            # 1. 从当前 gen_batch 取真实的 prompt
            in_ids = gen_batch.batch["input_ids"][idx].unsqueeze(0)
            
            resp_ids = dummy_resp
            old_logps = dummy_logp
            
        resp_mask = (resp_ids != pad_token_id).long()
        combined_ids = torch.cat([in_ids, resp_ids], dim=1)
        attn_mask = (combined_ids != pad_token_id).long()
        pos_ids = attn_mask.cumsum(dim=1) - 1
        pos_ids.masked_fill_(attn_mask == 0, 0)

        log_probs_list.append(old_logps)
        response_masks_list.append(resp_mask)
        prompts_list.append(in_ids)
        responses_list.append(resp_ids)
        position_ids_list.append(pos_ids)

    # 3. 拼接成完整的 [batch_size, seq_len] 张量
    return {
        "log_probs": torch.cat(log_probs_list, dim=0),
        "response_masks": torch.cat(response_masks_list, dim=0),
        "prompts": torch.cat(prompts_list, dim=0),
        "responses": torch.cat(responses_list, dim=0),
        "position_ids": torch.cat(position_ids_list, dim=0),
    }

class SpecDecodingPreRolloutResult:
    """Container for pre-rollout speculative decoding results."""

    __slots__ = (
        "gen_batch",
        "pad_size",
        "metrics",
        "have_pre_rollouts",
        # saved for post-rollout merge
        "prompt_ids",
        "prompt_attn_mask",
        "prompt_pos_ids",
        "aligned_old_responses",
        "aligned_old_response_mask",
        "cut_idx",
        "idx_reuse",
        "idx_need",
    )


class HisReuseManager:

    def __init__(
        self,
        trainer_config: Any,
        # 以下为外部依赖，也可以直接用 import；为方便测试/解耦建议显式传入
        actor_rollout_wg: Any,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        # 配置（按你原始配置取值，这里给默认语义）
        self.n_repeat: int = trainer_config.actor_rollout_ref.rollout.n
        self.save_rollout_dir: Optional[str] = trainer_config.trainer.get("rollout_data_dir", None)
        self.reuse_scale = trainer_config.trainer.get("reuse_scale", 0.0)
        self.num_workers = trainer_config.actor_rollout_ref.rollout.agent.num_workers

        # 外部组件
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
 
        # hash index  
        self.cache = SimpleFileCache(base_dir=trainer_config.trainer.get("rollout_data_dir", "./his_data"), chunk_size=1000)  

        # 状态：历史轮转（你原来的 latest_old_policy / latest_old_policy_tensor）
        self.latest_old_policy = defaultdict(list)
        self.latest_old_policy_tensor = defaultdict(list)

        # 确保目录存在（如果需要落盘）
        if self.save_rollout_dir:
            os.makedirs(self.save_rollout_dir, exist_ok=True)


    def pre_reuse_rl_generate_patch(self, 
        gen_batch: DataProto,
        async_rollout_manager: Any,
        async_rollout_mode: Any,
        global_steps: Any,       
        metrics: dict, 
        timing_raw: dict,
    ) -> Optional[SpecDecodingPreRolloutResult]:

        result = SpecDecodingPreRolloutResult()

        print(f"Begin Speculative Decoding...")
        
        from verl.utils.debug import marked_timer

        with marked_timer("pre_log_probs", timing_raw, color="blue"):

            cached_tensors = self.cache.query_batch(gen_batch.non_tensor_batch['raw_prompt'], n_repeat=self.n_repeat)

            if not cached_tensors:
                result.have_pre_rollouts = False
                return result
            
            result.have_pre_rollouts = True

            aligned = align_prev_to_gen(
                cached_tensors=cached_tensors,
                gen_batch=gen_batch,
                pad_token_id=151643,
            )

            aligned_old_logp = aligned["log_probs"]
            aligned_old_response_mask = aligned['response_masks']
            prompt_ids = aligned["prompts"]
            prompt_attn_mask = (prompt_ids != 151643).long()
            prompt_pos_ids = prompt_attn_mask.cumsum(dim=1) * prompt_attn_mask - prompt_attn_mask
            
            aligned_old_responses = aligned["responses"]
            aligned_position_ids = aligned["position_ids"]

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # prompt_ids = gen_batch.batch["input_ids"]  # [B,P]
            # prompt_attn_mask = gen_batch.batch["attention_mask"]  # [B,P]
            # prompt_pos_ids = gen_batch.batch["position_ids"]
            gen_batch.batch["input_ids"] = prompt_ids
            gen_batch.batch["attention_mask"] = prompt_attn_mask
            gen_batch.batch["position_ids"] = prompt_pos_ids

            attention_mask = torch.cat([prompt_attn_mask, aligned_old_response_mask], dim=1)
            input_ids = torch.cat([prompt_ids, aligned_old_responses], dim=-1)
            # import pdb; pdb.set_trace()
            
        
            pre_prob_data = DataProto.from_single_dict({
                "responses": aligned_old_responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": aligned_position_ids
            })
            pre_log_probs = self.actor_rollout_wg.compute_log_prob(pre_prob_data)
            old_logp = aligned_old_logp
            new_logp = pre_log_probs.batch['old_log_probs']
            # import pdb; pdb.set_trace()
            response_mask = aligned_old_response_mask

            if self.reuse_scale == 0.0:
                out = spec_cut(
                    old_logp=old_logp,
                    new_logp=new_logp,
                    response_mask=response_mask,
                    p_abs_thresh=None,
                    seed=1234,
                )
            else:
                print("=======enter spec_cut_with_knobs======")
                out = spec_cut_with_knobs(
                    old_logp=old_logp,
                    new_logp=new_logp,
                    response_mask=response_mask,
                    bias=self.reuse_scale if self.reuse_scale < 0 else -self.reuse_scale,
                    p_abs_thresh=None,
                    seed=1234,
                )

            cut_idx = out["cut_idx"]  # [B]
            idx_reuse = out["idx_reuse"]  # [Nr]
            idx_need = out["idx_need"]  # [Nn]

            rows = idx_need
            p_ids = gen_batch.batch["input_ids"][rows]
            p_msk = gen_batch.batch["attention_mask"][rows]
            # import pdb; pdb.set_trace()
            p_pos = gen_batch.batch["position_ids"][rows]
            ctx_ids, ctx_msk, ctx_pos, max_k = build_ctx(
                p_ids, p_msk, p_pos,
                aligned_old_responses[rows],
                cut_idx[rows],
                pad_id=tokenizer.pad_token_id,
            )

            need_dp = DataProto.from_single_dict({
                "input_ids": ctx_ids,
                "attention_mask": ctx_msk,
                "position_ids": ctx_pos,
            })
            need_dp.meta_info = dict(gen_batch.meta_info)
            
            per_req_np = out["per_request_max_new_tokens"][idx_need.cpu()].detach().cpu().numpy().astype(np.int32)

            # need_dp.meta_info["per_request_max_new_tokens"] = per_req
            need_dp.non_tensor_batch["per_request_max_new_tokens"] = per_req_np
            need_dp.non_tensor_batch["raw_prompt"] = gen_batch.non_tensor_batch["raw_prompt"][rows.cpu().numpy()]
            need_dp.non_tensor_batch["data_source"] = gen_batch.non_tensor_batch["data_source"][rows.cpu().numpy()]
            need_dp.non_tensor_batch["reward_model"] = gen_batch.non_tensor_batch["reward_model"][rows.cpu().numpy()]
            need_dp.meta_info["prefix_reuse"] = True

            # === Pad to DP world_size multiple ===
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not async_rollout_mode
                else self.num_workers
            )

            if idx_need.shape[0] > 0:
                need_dp_padded, pad_size = pad_dataproto_to_divisor(need_dp, size_divisor)
                gen_batch = need_dp_padded
            else:
                pad_size = None
                gen_batch = need_dp
        
        result.gen_batch = gen_batch
        result.pad_size = pad_size
        result.metrics = out["metrics"]
        result.prompt_ids = prompt_ids
        result.prompt_attn_mask = prompt_attn_mask
        result.prompt_pos_ids = prompt_pos_ids
        result.aligned_old_responses = aligned_old_responses
        result.aligned_old_response_mask = aligned_old_response_mask
        result.cut_idx = cut_idx
        result.idx_reuse = idx_reuse
        result.idx_need = idx_need

        return result


    def pre_reuse_decoding_post_rollout(
        self,
        gen_batch_output: DataProto,
        pre_result: SpecDecodingPreRolloutResult,
        timing_raw: dict,
    ) -> DataProto:       
        
        from verl.utils.debug import marked_timer

        with marked_timer("post-rollout", timing_raw, color="red"):
            if pre_result.idx_need.shape[0] > 0:
                gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pre_result.pad_size)
                gen_out_meta_info = gen_batch_output.meta_info
            else:
                gen_out_meta_info = {}

            prompt_ids = pre_result.prompt_ids
            prompt_attn_mask = pre_result.prompt_attn_mask
            prompt_pos_ids = pre_result.prompt_pos_ids
            aligned_old_responses = pre_result.aligned_old_responses
            aligned_old_response_mask = pre_result.aligned_old_response_mask
            cut_idx = pre_result.cut_idx
            idx_reuse = pre_result.idx_reuse
            idx_need = pre_result.idx_need

            pad_id = self.tokenizer.pad_token_id

            B, P = prompt_ids.shape
            R = aligned_old_responses.shape[1]

            # pre-allocate
            final_inputs = torch.full((B, P + R), pad_id, dtype=prompt_ids.dtype)
            final_attn = torch.zeros((B, P + R), dtype=prompt_attn_mask.dtype)
            final_pos = torch.zeros((B, P + R), dtype=prompt_pos_ids.dtype)
            final_resp = torch.full((B, R), pad_id, dtype=prompt_ids.dtype)

            # fill prompt segment
            final_inputs[:, :P] = prompt_ids
            final_attn[:, :P] = prompt_attn_mask
            final_pos[:, :P] = prompt_pos_ids

            # position increment constant
            delta = torch.arange(1, R + 1, dtype=prompt_pos_ids.dtype)

            if idx_reuse.numel() > 0:
                final_resp[idx_reuse] = aligned_old_responses[idx_reuse]
                final_attn[idx_reuse, P:P + R] = aligned_old_response_mask[idx_reuse]
                last_pos_reuse = prompt_pos_ids[idx_reuse, -1].unsqueeze(1)
                final_pos[idx_reuse, P:P + R] = last_pos_reuse + delta

            # trunc_total = 0
            trunc_nonpad = 0
            trunc_nonpad_tokens = 0
            if idx_need.numel() > 0:
                new_resp = gen_batch_output.batch["responses"]
                # row-wise copy (for loop is fast on CPU)
                for row_in_sub, i in enumerate(idx_need.tolist()):
                    k = int(cut_idx[i])  # prefix length that can be reused
                    keep_pref = min(k, R)  # prefix length cannot exceed R
                    take_new = R - keep_pref  # new response length that can be put

                    if keep_pref > 0:
                        final_resp[i, :keep_pref] = aligned_old_responses[i, :keep_pref]

                    if take_new > 0:
                        final_resp[i, keep_pref:keep_pref + take_new] = new_resp[row_in_sub, :take_new]

                    # 3) count the number of non-pad tokens in the truncated "new response"
                    trunc_len = min(k, R)
                    if trunc_len > 0:
                        tail = new_resp[row_in_sub, R - trunc_len: R]
                        # trunc_total   += trunc_len
                        trunc_nonpad_tokens += int((tail != pad_id).sum().item())
                        if trunc_nonpad_tokens > 0:
                            trunc_nonpad += 1

                # —— here reconstruct the response segment mask of need rows, **include the first pad/eot**
                nr = idx_need.long()
                need_resp_full = final_resp[nr]  # [N_need, R]
                is_pad = (need_resp_full == pad_id)  # [N_need, R]
                has_pad = is_pad.any(dim=1)  # [N_need]
                # position of the first pad (0 when no pad)
                first_pad_idx = torch.argmax(is_pad.to(torch.int32), dim=1)  # [N_need]
                # inclusive length: has pad -> idx+1; no pad -> R
                L_inclusive = torch.where(
                    has_pad,
                    first_pad_idx + 1,
                    torch.full_like(first_pad_idx, fill_value=need_resp_full.size(1))
                )  # [N_need]
                col = torch.arange(need_resp_full.size(1), dtype=prompt_attn_mask.dtype).unsqueeze(0)  # [1,R]
                resp_mask_need = (col < L_inclusive.unsqueeze(1)).to(prompt_attn_mask.dtype)  # [N_need,R]

                final_attn[nr, P:P + R] = resp_mask_need
                last_pos_need = prompt_pos_ids[nr, -1].unsqueeze(1)
                final_pos[nr, P:P + R] = last_pos_need + delta

                final_inputs[:, P:P + R] = final_resp

            # assemble final DataProto
            merged = {
                "prompts": prompt_ids,  # [B,P]
                "responses": final_resp,  # [B,R]
                "input_ids": final_inputs,  # [B,P+R]
                "attention_mask": final_attn,  # [B,P+R]
                "position_ids": final_pos,  # [B,P+R]
            }

            gen_batch_output = DataProto.from_single_dict(merged)
            gen_batch_output.meta_info = gen_out_meta_info

        return gen_batch_output, timing_raw




class SimpleFileCache:
    """
    极简哈希文件缓存：无索引、无淘汰、直接覆盖。
    目录结构: his_data/ab/cd/abcd1234...ef.pt
    """
    def __init__(self, base_dir: str = "./his_data", chunk_size: int = 1000):
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        # 启动时不需要做任何事情，没有 index 要读，没有文件要扫
        
        # 写入依然需要异步队列（防止同步写 64 个 .pt 卡住 GPU 训练几十毫秒）
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._write_worker, daemon=True)
        self._thread.start()

    def _get_path(self, h: str) -> str:
        """
        拿到哈希后，直接算出它属于哪个 chunk。
        取前 8 位十六进制 (32位整数) 对 chunk_size 取模，保证绝对均匀打散。
        """
        chunk_id = int(h[:8], 16) % self.chunk_size
        return os.path.join(self.base_dir, f"chunk_{chunk_id}", f"{h}.pt")

    # ==============================================================
    # 1. 查询：直接拼路径，exists 判断，load 读取
    # ==============================================================
    def query_batch(self, prompts: List[str], n_repeat: int = 1) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        返回格式与之前完全一致: { batch_idx: {"response_ids": tensor, ...} }
        """
        results = {}
        # 因为 repeat(interleave=True)，所以每隔 n_repeat 步才是新的 prompt
        num_unique_prompts = len(prompts) // n_repeat
        
        for i in range(num_unique_prompts):
            h = self._hash(prompts[i * n_repeat])
            pt_path = self._get_path(h)
            
            if os.path.exists(pt_path):
                try:
                    # 从磁盘读出的是一个包含 n_repeat 个元素的 List
                    data_list = torch.load(pt_path, map_location="cpu", weights_only=False)
                    
                    # 安全检查：如果之前存的是别的 n_repeat 数量，直接废弃不匹配的缓存
                    if isinstance(data_list, list) and len(data_list) == n_repeat:
                        # 将 list 里的 8 个元素，分别映射回 batch 里的 8 个 idx
                        for j in range(n_repeat):
                            batch_idx = i * n_repeat + j
                            results[batch_idx] = data_list[j]
                except Exception:
                    # 文件损坏等情况，静默跳过
                    pass
        return results

    # ==============================================================
    # 2. 写入：扔进队列，后台直接覆盖写
    # ==============================================================
    def update_batch_async(
        self,
        prompts: List[str],
        responses: torch.Tensor,
        input_ids: torch.Tensor,
        old_logps: torch.Tensor,
        n_repeat: int = 1,  # 新增参数
    ):
        batch_size = len(prompts)
        if batch_size == 0:
            return
            
        num_unique_prompts = batch_size // n_repeat
        
        for i in range(num_unique_prompts):

            prompt_str = prompts[i * n_repeat]

            grouped_data_list = []
            for j in range(n_repeat):
                idx = i * n_repeat + j
                grouped_data_list.append({
                    "response_ids": responses[idx].detach().cpu(),
                    "input_ids": input_ids[idx].detach().cpu(),
                    "old_logps": old_logps[idx].detach().cpu(),
                })
                
            # 作为一个整体 item 扔进队列
            self._queue.put({
                "prompt_str": prompt_str,
                "grouped_data_list": grouped_data_list
            })

    # ==============================================================
    # 内部机制
    # ==============================================================
    def _write_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break

            h = self._hash(item["prompt_str"])
            pt_path = self._get_path(h)
            
            # 2. 直接拿到 CPU 数据 (它本身就是我们需要存入的 List 格式)
            data_list_to_save = item["grouped_data_list"]
            
            # 3. 创建目录并落盘
            os.makedirs(os.path.dirname(pt_path), exist_ok=True)
            torch.save(data_list_to_save, pt_path)
            
        self._queue.task_done()

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=60)

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.md5(str(prompt).encode("utf-8")).hexdigest()
