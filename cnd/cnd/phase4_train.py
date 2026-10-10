"""Phase 4: 学生训练（FSDP 全分片，全 8 卡）。

arm:
- opd:      A_t = w_kl·ω_t·(−RKL_t)                纯蒸馏（免答案）
- opd_acc:  A_t = w_kl·ω_t·(−RKL_t) + λ·A_acc      规范形态（+ L_RL 项）

select.mode：
- causal_credit: 步级因果敏感度加权（phase2_intervene 的 cc.jsonl，φ_k）
- cnd:           步级分支 MC 必要性加权（phase2_branch 的 cnd.jsonl，N_k；消融臂）
- none:          全量均匀（baseline；ω ≡ 1）

单 epoch 纯 on-policy REINFORCE（epochs>1 启用存储 logprobs 的 PPO-clip）。
常因子在优势归一化之后施加（防 z-norm 抵消）。
"""
import argparse
import math
from pathlib import Path

import torch
import torch.distributed as dist

from .common import load_config, read_jsonl, round_dir, set_seed, write_jsonl
from .weights import causal_credit_weights, cnd_step_weights, omega_advantage


def build_advantage(cfg, resp):
    """mode=none 的全量基线优势：A_t = w_kl·(−RKL_t)（+ λ·A_acc）。"""
    t = cfg["train"]
    rkl = torch.tensor(resp["rkl"], dtype=torch.float32)
    adv = -rkl
    if t["adv_norm"]:
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    adv = t["w_kl"] * adv
    sc = t.get("select", {}).get("correct_scale", 1.0)
    if sc != 1.0 and resp.get("correct"):
        adv = sc * adv
    if t["arm"] == "opd_acc":
        adv = adv + t["lambda_acc"] * float(resp["acc_adv"])
    return adv


_SIGNAL_FILE = {"cnd": "cnd.jsonl", "causal_credit": "cc.jsonl"}


def load_signal(cfg, k, mode):
    """(qid, resp_idx) -> {step_idx: 标量}（phase2 落盘：cnd=N_k / cc=φ_k）。"""
    try:
        rows = read_jsonl(round_dir(cfg["work_dir"], k) / _SIGNAL_FILE[mode])
    except FileNotFoundError:
        return {}
    key = "necessity" if mode == "cnd" else "phi"
    return {(r["qid"], r["resp"]): r[key] for r in rows}


def load_round_samples(cfg, k):
    t = cfg["train"]
    rows = read_jsonl(round_dir(cfg["work_dir"], k) / "graded.jsonl")
    sc = t.get("select", {})
    mode = sc.get("mode", "causal_credit")
    sig_map = load_signal(cfg, k, mode) if mode in _SIGNAL_FILE else {}
    samples = []
    agg = {"frac": [], "kl_cov": [], "dropped": 0}
    for r in rows:
        for gi, resp in enumerate(r["responses"]):
            n = len(resp["rkl"])
            if n == 0:
                continue
            if mode in ("cnd", "causal_credit"):
                spans = resp.get("step_spans") or []
                sd = sig_map.get((r["qid"], gi))
                if sd and spans:
                    vals = [sd.get(str(i), sd.get(i)) for i in range(len(spans))]
                    if mode == "causal_credit":
                        omega, _ = causal_credit_weights(vals, spans, sc.get("alpha", 1.0))
                    else:
                        omega, _ = cnd_step_weights(vals, spans, sc.get("alpha", 1.0))
                else:
                    omega = [1.0] * n  # 信号缺失回退：均匀（= 全量 OPSD + acc）
                omega = (omega + [0.0] * n)[:n]
                if not any(x > 0 for x in omega):
                    agg["dropped"] += 1
                    continue
                adv, mask = omega_advantage(
                    resp["rkl"], omega, w_kl=t["w_kl"], adv_norm=t["adv_norm"],
                    correct=resp.get("correct", False),
                    correct_scale=sc.get("correct_scale", 1.0),
                    acc_adv=resp.get("acc_adv"),
                    lambda_acc=t["lambda_acc"] if t["arm"] == "opd_acc" else 0.0)
                resp["sel_mask"] = mask
                agg["frac"].append(sum(mask) / n)
                agg["kl_cov"].append(
                    sum(k for k, m in zip(resp["rkl"], mask) if m)
                    / (sum(resp["rkl"]) or 1.0))
            else:  # none：全量均匀
                mask = [True] * n
                resp["sel_mask"] = mask
                adv = build_advantage(cfg, resp)
                agg["frac"].append(1.0)
                agg["kl_cov"].append(1.0)
            samples.append({"question": r["question"], "token_ids": resp["token_ids"],
                            "chosen_lp": resp["chosen_lp"], "sel_mask": resp["sel_mask"],
                            "adv": adv})
    ns = max(len(agg["frac"]), 1)
    agg["frac"] = sum(agg["frac"]) / ns
    agg["kl_cov"] = sum(agg["kl_cov"]) / ns
    return samples, agg


def find_layer_classes(model):
    seen, out = set(), []
    for name, mod in model.named_modules():
        cls = type(mod)
        if cls.__name__ in seen:
            continue
        seen.add(cls.__name__)
        if "DecoderLayer" in cls.__name__ or "Blocks" in cls.__name__:
            out.append(cls)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]
    set_seed(t["seed"])

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["student_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"], torch_dtype=torch.float32,
        attn_implementation="flash_attention_2")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    layer_cls = find_layer_classes(model)
    assert layer_cls, "未找到 DecoderLayer 类，检查模型架构"
    fsdp = FSDP(
        model, auto_wrap_policy=lambda *a, **kw: transformer_auto_wrap_policy(
            *a, transformer_layer_cls=layer_cls, **kw),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.bfloat16,
                                       buffer_dtype=torch.bfloat16),
        device_id=torch.cuda.current_device(), use_orig_params=True,
        limit_all_gathers=True,
    )
    opt = torch.optim.AdamW(fsdp.parameters(), lr=t["lr"],
                            betas=tuple(t["betas"]), weight_decay=t["weight_decay"])

    samples, sel_stats = load_round_samples(cfg, args.round)
    if rank == 0:
        print(f"[train] round {args.round} arm={t['arm']}: {len(samples)} samples")
        print(f"[train] select mode={t.get('select', {}).get('mode', 'causal_credit')}: "
              f"frac={sel_stats['frac']:.3f} kl_cov={sel_stats['kl_cov']:.3f} "
              f"dropped_seqs={sel_stats['dropped']}")

    # ---- 构造批次 ----
    mb = t["micro_batch"]
    rng = torch.Generator().manual_seed(t["seed"] + args.round)
    order = torch.randperm(len(samples), generator=rng).tolist()
    order = order[rank::world]  # 按 rank 分片：有效 batch = mb·grad_accum·world
    steps_per_epoch = max(1, math.ceil(len(order) / (mb * t["grad_accum"])))
    warm = min(t["warmup_steps"], steps_per_epoch)
    lr_fn = lambda s: t["lr"] * min((s + 1) / max(warm, 1),
                                    0.5 * (1 + math.cos(math.pi * s / steps_per_epoch)))

    fsdp.train()
    stats = {"loss": 0.0, "n": 0}
    opt.zero_grad(set_to_none=True)
    step = 0
    micro = []
    for idx in order:
        micro.append(samples[idx])
        if len(micro) < mb:
            continue

        pids = [tok.apply_chat_template(
            [{"role": "user", "content": s["question"]}], add_generation_prompt=True)
            for s in micro]
        maxlen = max(len(p) + len(s["token_ids"]) for p, s in zip(pids, micro))
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(micro), maxlen), pad, dtype=torch.long)
        resp_mask = torch.zeros((len(micro), maxlen), dtype=torch.bool)
        adv_t = torch.zeros((len(micro), maxlen), dtype=torch.float32)
        old_lp = torch.zeros((len(micro), maxlen), dtype=torch.float32)
        for i, (p, s) in enumerate(zip(pids, micro)):
            seq = p + list(s["token_ids"])
            n = min(len(seq), maxlen)
            ids[i, :n] = torch.tensor(seq[:n])
            rs = min(len(p), n)
            # 响应区只放行入选 token（未入选位置 loss 不计）
            resp_mask[i, rs:n] = torch.tensor(
                [bool(x) for x in s["sel_mask"][:n - rs]], dtype=torch.bool)
            a = s["adv"]
            adv_t[i, rs:rs + len(a)] = a[:n - rs]
            ol = s["chosen_lp"]
            old_lp[i, rs:rs + len(ol)] = torch.tensor(ol[:n - rs])

        ids = ids.cuda(rank)
        resp_mask = resp_mask.cuda(rank)
        logits = fsdp(input_ids=ids).logits.float()
        logp_all = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
        m = resp_mask[:, 1:].float()

        adv = adv_t[:, 1:].cuda(rank)
        old = old_lp[:, 1:].cuda(rank)
        if t["epochs"] > 1:
            ratio = (tok_logp - old).exp()
            s1 = ratio * adv
            s2 = ratio.clamp(1 - t["clip_eps"], 1 + t["clip_eps"]) * adv
            per_tok = torch.min(s1, s2)
        else:  # 纯 on-policy 单 epoch：直接 REINFORCE
            per_tok = tok_logp * adv
        loss = -(per_tok * m).sum() / m.sum().clamp(min=1)

        (loss / t["grad_accum"]).backward()
        stats["loss"] += loss.item()
        stats["n"] += 1

        if stats["n"] % t["grad_accum"] == 0:
            for g in opt.param_groups:
                g["lr"] = lr_fn(step)
            torch.nn.utils.clip_grad_norm_(fsdp.parameters(), t["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % 5 == 0:
                print(f"[train] step {step}/{steps_per_epoch} loss={stats['loss']/stats['n']:.4f}")
            if step >= steps_per_epoch:
                break
        micro = []

    # ---- 保存（全量 state dict，rank0 落盘）----
    dist.barrier()
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg_sd):
        sd = fsdp.state_dict()
    if rank == 0:
        out_dir = Path(t.get("ckpt_dir", "./ckpt"))
        out_dir.mkdir(parents=True, exist_ok=True)
        fsdp.module.config.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        from safetensors.torch import save_file
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        write_jsonl(Path(cfg["work_dir"]) / "train_log.jsonl",
                    [{"round": args.round, "loss": stats["loss"] / max(stats["n"], 1),
                      "steps": step, "sel_frac": sel_stats["frac"],
                      "kl_cov": sel_stats["kl_cov"]}])
        print(f"[train] round {args.round} saved -> {out_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
