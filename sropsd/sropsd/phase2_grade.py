"""Phase 2: 教师打分 —— 学生 rollout 上逐 token 四信号。

每条响应落盘（graded.jsonl）：
- rkl:      KL(p_S‖p_T) 精确值（online logsumexp 分块，防溢出）
- entropy:  学生熵 H(p_S)（token 级 dual 选择的轴二代用品）
- ier:      梯度信噪比 IER = ||g||²/E||ĝ||²（SR-OPSD 轴二）。精确 KL 梯度
            g = p_S⊙(ℓ−KL)（ℓ=逐词表 log-ratio），其单样本 score-function
            估计 ĝ(y)=ℓ_y(e_y−p_S) 无偏；IER = ||g||²/(||g||²+Var)，
            在学生/教师 top-K 并集上近似。
- logratio: 采样 token 上 log π_S − log π_T（SR-OPSD 轴一 U_k 的原料）
另落盘 step_spans（token 步骤边界）。

teacher_sees_gold=true 时教师走独立 prompt（注入参考答案 y*，自蒸馏设定
π_T(·|x,y*,·)）：学生/教师序列分开前向、按各自 resp_start 对齐。跨模型使用
要求两模型 tokenizer 一致（Qwen 家族满足）。
"""
import argparse
from collections import defaultdict

import torch

from .common import (
    latest_student, load_config, read_jsonl, round_dir, score_answer,
    token_step_spans, write_jsonl,
)
from .prompts import qa_messages, qa_messages_gold


def load_pair(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tdev = cfg["grade"]["teacher_devices"]
    sdev = cfg["grade"]["student_devices"]
    tmem = {d: "22GiB" for d in tdev}
    smem = {d: "22GiB" for d in sdev}
    print(f"[grade] teacher -> {tdev}, student -> {sdev}")

    student_path = latest_student(cfg)  # 当前学生（round>0 时为 ckpt）
    tok = AutoTokenizer.from_pretrained(student_path)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"], torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=tmem, max_memory=tmem)
    teacher.eval()
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=smem, max_memory=smem)
    student.eval()
    return tok, teacher, student


def _streaming_lse(logits, lo, hi, device, chunk):
    """online logsumexp：逐块维护 running (max, sum)，防溢出。返回 [R,1]。"""
    m = None
    s = None
    V = logits.shape[-1]
    for start in range(0, V, chunk):
        l = logits[lo:hi, start:start + chunk].to(device)
        cm = l.max(-1, keepdim=True).values
        if m is None:
            m, s = cm, l.sub(cm).exp().sum(-1, keepdim=True)
        else:
            nm = torch.maximum(m, cm)
            s = s.mul((m - nm).exp()) + l.sub(nm).exp().sum(-1, keepdim=True)
            m = nm
    return m + s.log()


def build_seq(prompt_ids, resp_ids, max_seq_len):
    """prompt + 响应截断（保响应尾部），返回 (ids tensor, resp_start)。"""
    seq = list(prompt_ids) + list(resp_ids)
    if len(seq) > max_seq_len:
        seq = seq[-max_seq_len:]
        rs = max(1, len(seq) - len(resp_ids))
    else:
        rs = len(prompt_ids)
    return torch.tensor(seq), rs


@torch.no_grad()
def response_signal(student, teacher, ids_s, rs_s, ids_t, rs_t,
                    vocab_chunk, ier_topk=32):
    """对响应区每个位置计算 (RKL, H(p_S), IER, log-ratio)，fp16 cpu 各 [R]。"""
    dev = student.device
    R = len(ids_s) - rs_s
    assert R > 0 and len(ids_t) - rs_t == R, "响应区长度不一致"

    ls_all = student(input_ids=ids_s.unsqueeze(0).to(dev), use_cache=False).logits[0].float()
    lt_all = teacher(input_ids=ids_t.unsqueeze(0).to(teacher.device),
                     use_cache=False).logits[0].float()

    lo_s, hi_s = rs_s - 1, len(ids_s) - 1
    lo_t, hi_t = rs_t - 1, len(ids_t) - 1
    V = ls_all.shape[-1]

    lse_s = _streaming_lse(ls_all, lo_s, hi_s, dev, vocab_chunk)
    lse_t = _streaming_lse(lt_all, lo_t, hi_t, teacher.device, vocab_chunk)

    # 全词表 log p_S 一次算好（RKL/熵/采样 token logprob/IER 共用）
    ls_full = ls_all[lo_s:hi_s] - lse_s          # [R, V]

    # 1) RKL 与学生熵（教师侧按块搬运，控峰值显存）
    rkl = None
    ent = None
    for start in range(0, V, vocab_chunk):
        end = min(start + vocab_chunk, V)
        ls = ls_full[:, start:end]
        lt = (lt_all[lo_t:hi_t, start:end] - lse_t).to(dev)
        p_s = ls.exp()
        part = (p_s * (ls - lt)).sum(-1)          # [R]
        rkl = part if rkl is None else rkl + part
        e_part = -(p_s * ls).sum(-1)
        ent = e_part if ent is None else ent + e_part

    # 2) 采样 token 的师生 log-ratio
    tgt = ids_s[lo_s + 1:hi_s + 1].to(dev)        # [R] 响应 token
    lp_s = ls_full.gather(1, tgt.view(-1, 1)).squeeze(1)
    lp_t = (lt_all[lo_t:hi_t] - lse_t).gather(
        1, tgt.to(teacher.device).view(-1, 1)).squeeze(1).to(dev)
    logratio = lp_s - lp_t

    # 3) IER：top-K 并集上的梯度信噪比
    idx_s = ls_all[lo_s:hi_s].topk(ier_topk, dim=-1).indices
    idx_t = lt_all[lo_t:hi_t].topk(ier_topk, dim=-1).indices
    u = torch.cat([idx_s, idx_t.to(dev)], dim=-1).sort(dim=-1).values
    dup = torch.zeros_like(u, dtype=torch.bool)
    dup[:, 1:] = u[:, 1:] == u[:, :-1]
    keep = (~dup).float()
    lt_u = (lt_all[lo_t:hi_t] - lse_t).gather(1, u.to(teacher.device)).to(dev)
    l_u = ls_full.gather(1, u) - lt_u             # log-ratio @ 并集
    p_u = ls_full.gather(1, u).exp()
    g2 = (keep * p_u.pow(2) * (l_u - rkl.unsqueeze(-1)).pow(2)).sum(-1)
    sump2 = (keep * p_u.pow(2)).sum(-1, keepdim=True)
    eg2 = (keep * p_u * l_u.pow(2) * (1.0 - 2.0 * p_u + sump2)).sum(-1)
    ier = (g2 / (eg2 + 1e-12)).clamp(0.0, 1.0)    # = ||g||²/E||ĝ||²

    return (rkl.clamp(min=0).half().cpu(), ent.clamp(min=0).half().cpu(),
            ier.half().cpu(), logratio.half().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc, gr = cfg["rounds"], cfg["grade"]
    gold_hint = bool(gr.get("teacher_sees_gold", False))

    tok, teacher, student = load_pair(cfg)
    rows = read_jsonl(round_dir(cfg["work_dir"], args.round) / "rollouts.jsonl")

    # 组内相对答案优势（同题 G 条响应，GRPO 式）
    group_correct = defaultdict(list)
    for r in rows:
        for resp in r["responses"]:
            ok = bool(resp["pred"]) and score_answer(resp["pred"], r["gold"])
            resp["correct"] = ok
            group_correct[r["qid"]].append(ok)

    out = []
    n_done = 0
    for r in rows:
        golds = group_correct[r["qid"]]
        mu = sum(golds) / len(golds)
        std = (sum((x - mu) ** 2 for x in golds) / len(golds)) ** 0.5
        acc_adv = 0.0 if std < 1e-6 else [(x - mu) / std for x in golds]

        ps = tok.apply_chat_template(qa_messages(r["question"]),
                                     add_generation_prompt=True)
        pt = tok.apply_chat_template(
            qa_messages_gold(r["question"], r["gold"]) if gold_hint
            else qa_messages(r["question"]),
            add_generation_prompt=True)
        vresps = []
        for resp, a_adv in zip(r["responses"], acc_adv):
            resp_ids = resp["token_ids"]
            ids_s, rs_s = build_seq(ps, resp_ids, rc["max_seq_len"])
            ids_t, rs_t = build_seq(pt, resp_ids, rc["max_seq_len"])
            rkl, ent, ier, logratio = response_signal(
                student, teacher, ids_s, rs_s, ids_t, rs_t,
                gr["vocab_chunk"], gr.get("ier_topk", 32))
            spans = token_step_spans(tok, ids_s[rs_s:].tolist(),
                                     window=gr.get("step_window", 64))
            lps = resp["chosen_lp"][-rkl.shape[0]:]
            vresps.append({
                "token_ids": ids_s[rs_s:].tolist(),
                "chosen_lp": [x for x in lps if x is not None],
                "rkl": rkl.tolist(),
                "entropy": [round(float(x), 4) for x in ent.tolist()],
                "ier": [round(float(x), 4) for x in ier.tolist()],
                "logratio": [round(float(x), 4) for x in logratio.tolist()],
                "step_spans": spans,
                "correct": resp["correct"],
                "acc_adv": a_adv,
            })
            n_done += 1
            if n_done % 100 == 0:
                print(f"[grade] {n_done} responses")
        out.append({"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                    "responses": vresps})

    write_jsonl(round_dir(cfg["work_dir"], args.round) / "graded.jsonl", out)
    mrkl = sum(sum(sum(rr["rkl"]) / max(len(rr["rkl"]), 1) for rr in r["responses"]) / len(r["responses"]) for r in out) / max(len(out), 1)
    mier = sum(sum(sum(rr["ier"]) / max(len(rr["ier"]), 1) for rr in r["responses"]) / len(r["responses"]) for r in out) / max(len(out), 1)
    print(f"[grade] round {args.round}: mean per-token RKL = {mrkl:.4f}, "
          f"mean IER = {mier:.4f} -> graded.jsonl")


if __name__ == "__main__":
    main()
