"""Phase 0: 训练难度探针 —— 用基座学生 pass@G' 把训练池过滤到中难度带。

中难度带（pass@G' ∈ [keep]）保证三件事：
- 答案组内优势 acc_adv 有方差（全对/全错组的组内优势恒 0，白算）；
- 蒸馏信号有区分度（太易的题师生处处对齐，无弥散度结构；太难的题
  全错 rollout 的步骤无"可教性"）；
- 粒度路由两种分支都有样本（全对齐 → 全 token 级；全弥散 → 全步骤级，
  路由退化为固定粒度）。
探针只用基座模型跑一次（训练期间不复用）；产出 prompts.jsonl + 难度分布报告。
probe.enable=false 时退化为全量落盘（旧行为）。
"""
import argparse
from collections import Counter

from .common import (
    build_llm, chat_many, extract_answer, load_config, score_answer, write_jsonl,
)
from .data import load_train
from .prompts import qa_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg.get("data", {})
    pb = d.get("probe", {})
    if not pb.get("enable", False):
        from .data import build_prompts
        build_prompts(cfg)  # 不过探：全量落盘
        return

    G = pb.get("G", 4)
    lo, hi = pb.get("keep", [1, max(G - 1, 1)])

    rows = load_train(cfg)
    n_empty = sum(1 for r in rows if not r.get("gold"))
    rows = [r for r in rows if r.get("gold")]  # 空 gold 判分恒 False，剔除
    print(f"[probe] pool={len(rows)} (dropped empty-gold {n_empty}), "
          f"G'={G}, keep pass@G' in [{lo},{hi}]")

    llm = build_llm(cfg["student_model"], tp=1,
                    max_model_len=cfg["rounds"]["max_seq_len"] + 64)
    msgs = [qa_messages(r["question"]) for r in rows]
    outs = chat_many(llm, msgs, n=G, temperature=pb.get("temperature", 0.8),
                     top_p=0.95, max_tokens=pb.get("max_tokens", 1024))

    kept = []
    hist = Counter()
    for r, comps in zip(rows, outs):
        n_ok = sum(int(score_answer(extract_answer(c.text), r["gold"])) for c in comps)
        hist[n_ok] += 1
        if lo <= n_ok <= hi:
            kept.append(r)

    max_n = d.get("max_questions", 0)
    if max_n and len(kept) > max_n:
        import random
        random.Random(cfg["train"]["seed"]).shuffle(kept)
        kept = kept[:max_n]

    write_jsonl(f"{cfg['work_dir']}/prompts.jsonl", kept)
    total = sum(hist.values())
    dist = " ".join(f"{k}:{hist[k] / total:.2f}" for k in sorted(hist))
    print(f"[probe] pass@G' histogram (0..{G}): {dist}")
    print(f"[probe] kept {len(kept)} mid-difficulty questions -> prompts.jsonl")


if __name__ == "__main__":
    main()
