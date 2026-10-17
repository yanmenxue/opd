"""Phase 2: 教师打分 —— 学生 rollout 上逐 token 师生对数比（GAM 的原料信号）。

每条响应落盘（graded.jsonl）：
- a: 逐 token 对数比 a_t = log π_T(y_t | x, y*, ŷ_<t) − log π_S(y_t | ŷ_<t)。
     E_{y~π_S}[a_t] = −KL_t：a 的均值承载蒸馏压力，符号携带方向
     （>0 = 教师更确信采样 token → 应提升）；其窗口内方差结构承载信用
     弥散信息（route.py 据此做粒度路由）。
另落盘 step_spans（token 步骤边界）+ 组内答案优势（GRPO 式）。

teacher_sees_gold=true 时教师走独立 prompt（注入参考答案 y*，自蒸馏设定
π_T(·|x,y*,·)）：学生/教师序列分开前向、按各自 resp_start 对齐。跨模型使用
要求两模型 tokenizer 一致（Qwen 家族满足）。
"""
import argparse
from collections import defaultdict

import torch

from .common import (
    chat_template_ids, latest_student, load_config, read_jsonl, round_dir,
    score_answer, token_step_spans, write_jsonl,
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
        attn_implementation="flash_attention_2",
        device_map="balanced", max_memory=tmem)  # device_map 只接受层映射/"auto"/"balanced"；
    teacher.eval()                               # 模型落在哪几张卡由 max_memory 的设备子集决定
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="balanced", max_memory=smem)
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
def response_logratio(student, teacher, ids_s, rs_s, ids_t, rs_t, vocab_chunk):
    """响应区每个位置的 a_t = log π_T(y_t) − log π_S(y_t)。返回 fp16 cpu [R]。"""
    dev = student.device
    R = len(ids_s) - rs_s
    assert R > 0 and len(ids_t) - rs_t == R, "响应区长度不一致"

    ls_all = student(input_ids=ids_s.unsqueeze(0).to(dev), use_cache=False).logits[0].float()
    lt_all = teacher(input_ids=ids_t.unsqueeze(0).to(teacher.device),
                     use_cache=False).logits[0].float()

    lo_s, hi_s = rs_s - 1, len(ids_s) - 1
    lo_t, hi_t = rs_t - 1, len(ids_t) - 1

    tgt = ids_s[lo_s + 1:hi_s + 1].to(dev)        # [R] 响应 token
    lse_s = _streaming_lse(ls_all, lo_s, hi_s, dev, vocab_chunk)
    lse_t = _streaming_lse(lt_all, lo_t, hi_t, teacher.device, vocab_chunk)

    lp_s = ls_all[lo_s:hi_s].gather(1, tgt.view(-1, 1)).squeeze(1) - lse_s.squeeze(-1)
    lp_t = (lt_all[lo_t:hi_t] - lse_t).gather(
        1, tgt.to(teacher.device).view(-1, 1)).squeeze(1).to(dev)
    return (lp_t - lp_s).half().cpu()


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
        acc_adv = [0.0] * len(golds) if std < 1e-6 else [(x - mu) / std for x in golds]

        ps = chat_template_ids(tok, qa_messages(r["question"]))
        pt = chat_template_ids(tok, qa_messages_gold(r["question"], r["gold"])
                               if gold_hint else qa_messages(r["question"]))
        vresps = []
        for resp, a_adv in zip(r["responses"], acc_adv):
            resp_ids = resp["token_ids"]
            ids_s, rs_s = build_seq(ps, resp_ids, rc["max_seq_len"])
            ids_t, rs_t = build_seq(pt, resp_ids, rc["max_seq_len"])
            a = response_logratio(student, teacher, ids_s, rs_s, ids_t, rs_t,
                                  gr["vocab_chunk"])
            spans = token_step_spans(tok, ids_s[rs_s:].tolist(),
                                     window=gr.get("step_window", 64))
            lps = resp["chosen_lp"][-a.shape[0]:]
            vresps.append({
                "token_ids": ids_s[rs_s:].tolist(),
                "chosen_lp": [x for x in lps if x is not None],
                "a": [round(float(x), 4) for x in a.tolist()],
                "step_spans": spans,
                "correct": resp["correct"],
                "acc_adv": a_adv,
                "boxed": "\\boxed" in (resp.get("text") or ""),
            })
            n_done += 1
            if n_done % 100 == 0:
                print(f"[grade] {n_done} responses")
        out.append({"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                    "responses": vresps})

    write_jsonl(round_dir(cfg["work_dir"], args.round) / "graded.jsonl", out)
    ma = sum(sum(sum(rr["a"]) / max(len(rr["a"]), 1) for rr in r["responses"]) / len(r["responses"]) for r in out) / max(len(out), 1)
    print(f"[grade] round {args.round}: mean per-token a = {ma:.4f} "
          f"(E[a]=-KL, 应 ≤ 0) -> graded.jsonl")


if __name__ == "__main__":
    main()
