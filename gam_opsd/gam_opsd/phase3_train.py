"""Phase 3: 学生训练（FSDP 全分片，全 8 卡）—— L_GAM 混合粒度蒸馏。

arm:
- opd:      L_GAM（免答案）
- opd_acc:  L_GAM + λ·A_acc                              规范形态（+ L_RL 项）
（RLVR/纯 GRPO baseline = train.w_kl: 0.0 + arm opd_acc）

route.mode 三档（route.py）：
- dynamic:  GAM——Disp_t > τ 的步骤走步骤级 A_step=Σa（求和），其余 token 级
- token:    全 token 级（= 标准 OPSD 主对照臂）
- step:     全步骤级（= 全步骤级 OPSD 主对照臂；固定粒度消融共用此开关）

单 epoch 纯 on-policy REINFORCE（epochs>1 启用存储 logprobs 的 PPO-clip）。
每轮路由诊断（弥散度分布 / 步级占比 / 弥散度×步长相关）追加 gam_diag.jsonl，
训练日志追加 train_log.jsonl。
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from .common import (
    append_jsonl, chat_template_ids, latest_student, load_config, read_jsonl,
    round_dir, set_seed,
)
from .route import gam_advantage


def load_round_samples(cfg, k):
    """graded.jsonl → 训练样本；聚合轮级路由诊断。"""
    t, r = cfg["train"], cfg.get("route", {})
    mode = r.get("mode", "dynamic")
    rows = read_jsonl(round_dir(cfg["work_dir"], k) / "graded.jsonl")
    samples = []
    infos = []
    n_drop = 0
    n_drop_unboxed = 0
    drop_unboxed = t.get("drop_unboxed", True)  # 未框答案的响应判分不可信（回退规则蒙的），不进训练池
    for row in rows:
        for resp in row["responses"]:
            if drop_unboxed and not resp.get("boxed", True):
                n_drop_unboxed += 1
                continue
            n = len(resp.get("a") or [])
            if n == 0:
                n_drop += 1
                continue
            adv, mask, info = gam_advantage(
                resp["a"], resp.get("step_spans"),
                mode=mode, tau=r.get("tau", 0.5), window=r.get("window", 16),
                eps=r.get("eps", 1e-6), w_kl=t["w_kl"],
                lambda_step=r.get("lambda_step", 1.0),
                adv_norm=r.get("adv_norm", False),
                correct=resp.get("correct", False),
                correct_scale=r.get("correct_scale", 1.0),
                acc_adv=resp.get("acc_adv"),
                lambda_acc=t["lambda_acc"] if t["arm"] == "opd_acc" else 0.0)
            if not any(x != 0.0 for x in adv):
                n_drop += 1  # 零优势（如 w_kl=0 且全对/全错组）无梯度
                continue
            resp["sel_mask"] = mask
            infos.append(info)
            samples.append({"question": row["question"], "token_ids": resp["token_ids"],
                            "chosen_lp": resp["chosen_lp"], "sel_mask": resp["sel_mask"],
                            "adv": adv})
    diag = {"n_responses": len(infos), "dropped": n_drop, "dropped_unboxed": n_drop_unboxed}
    for key in infos[0] if infos else ():
        vals = [i[key] for i in infos if i.get(key) is not None]
        diag[key] = sum(vals) / len(vals) if vals else None
    return samples, diag


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
    student_path = latest_student(cfg)  # round≥1 从上一轮 ckpt 续训，而非初始基座
    tok = AutoTokenizer.from_pretrained(student_path)
    model = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.float32,
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

    samples, diag = load_round_samples(cfg, args.round)
    if rank == 0:
        rt = cfg.get("route", {})
        print(f"[train] round {args.round} arm={t['arm']}: {len(samples)} samples")
        print(f"[train] route mode={rt.get('mode', 'dynamic')} tau={rt.get('tau', 0.5)} "
              f"window={rt.get('window', 16)}: frac_step_tokens={diag.get('frac_step_tokens')} "
              f"frac_steps={diag.get('frac_steps_routed')} n_steps={diag.get('n_steps')}")
        if diag.get("disp_mean") is not None:
            print(f"[train] dispersion mean={diag['disp_mean']:.3f} "
                  f"p25={diag.get('disp_p25')} median={diag['disp_median']:.3f} "
                  f"p75={diag.get('disp_p75')}")
        print(f"[train] dropped={diag.get('dropped')} 其中未框答案={diag.get('dropped_unboxed')}")

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

        pids = [chat_template_ids(tok, [{"role": "user", "content": s["question"]}])
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
            resp_mask[i, rs:n] = torch.tensor(
                [bool(x) for x in s["sel_mask"][:n - rs]], dtype=torch.bool)
            a = s["adv"]
            adv_t[i, rs:rs + len(a)] = torch.tensor(a[:n - rs], dtype=torch.float32)
            ol = s["chosen_lp"]
            old_lp[i, rs:rs + len(ol)] = torch.tensor(ol[:n - rs])

        ids = ids.cuda(rank)
        resp_mask = resp_mask.cuda(rank)
        tgt = ids[:, 1:]
        # 融合 CE 取选中 token 的 logp（log_softmax.gather ≡ -CE）。
        # cross_entropy 的 backward 需要 forward 的 fp32 log_softmax（[T,V]，
        # V=248320 时 T=5120 就 ~4.7GiB），再叠加 bf16 logits 及其梯度，
        # 是原 OOM 的主因。这里两层处理：
        #   1) 按序列分块（每块 fp32 ≤256MiB），避免整份 [T,V] 同时驻留；
        #   2) 每块用 checkpoint：backward 时重算该块的 fp32 softmax，
        #      算完即释放，保存值只剩 1 块（~0.25GiB）而不是整条序列。
        # 各位置独立，结果与整份计算逐位等价（只省显存，不改数值）。
        def _neg_ce(lg, tg, vocab):
            return -torch.nn.functional.cross_entropy(
                lg.float().reshape(-1, vocab), tg.reshape(-1),
                reduction="none")

        logits = fsdp(input_ids=ids).logits                              # [B, T, V] bf16
        V, n_out = logits.size(-1), ids.size(1) - 1
        chunk = max(1, (256 << 20) // (4 * V))
        parts = []
        for s in range(0, n_out, chunk):
            e = min(s + chunk, n_out)
            ce = checkpoint(_neg_ce, logits[:, s:e], tgt[:, s:e], V,
                            use_reentrant=False)
            parts.append(ce.view(ids.size(0), e - s))
        tok_logp = torch.cat(parts, dim=1)                                # [B, T-1]
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
        # ckpt 只含文本骨干权重，必须声明为纯文本 CausalLM 架构（原始模型是
        # VL/多模态时尤其如此，否则 vLLM 会走多模态加载路径）。
        # 显式配置优先；否则回退原始 config 的声明
        t_arch = t.get("ckpt_architectures")
        t_mtype = t.get("ckpt_model_type")
        if not (t_arch and t_mtype):
            with open(Path(cfg["student_model"]) / "config.json", encoding="utf-8") as f:
                orig_meta = json.load(f)
            t_arch = t_arch or orig_meta.get("architectures")
            t_mtype = t_mtype or orig_meta.get("model_type")
        if isinstance(t_arch, str):  # transformers v5 严格校验：architectures 须为 list[str]
            t_arch = [t_arch]
        fsdp.module.config.architectures = t_arch
        fsdp.module.config.model_type = t_mtype
        fsdp.module.config.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        from safetensors.torch import save_file
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        # 评测轮留一份快照：ckpt 每轮被覆盖，事后无法换预算复评（见 rounds.eval_every）
        ev = cfg["rounds"].get("eval_every", 0)
        if ev and (args.round + 1) % ev == 0:
            import shutil
            snap = out_dir / f"round_{args.round:03d}"
            snap.mkdir(parents=True, exist_ok=True)
            for f in out_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, snap / f.name)
            print(f"[train] 快照 -> {snap}")
        append_jsonl(Path(cfg["work_dir"]) / "train_log.jsonl",
                     {"round": args.round, "loss": stats["loss"] / max(stats["n"], 1),
                      "steps": step, "n_samples": len(samples), **diag})
        append_jsonl(Path(cfg["work_dir"]) / "gam_diag.jsonl",
                     {"round": args.round, **diag})
        print(f"[train] round {args.round} saved -> {out_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
