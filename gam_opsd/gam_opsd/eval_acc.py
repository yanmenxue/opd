"""准确率评测：vLLM 贪心解码 + math_verify；MATH-500 按 level 分层报告。"""
import argparse
from collections import defaultdict

from .common import (
    build_llm, chat_many, extract_answer, latest_student, load_config, score_answer,
    write_jsonl,
)
from .data import load_eval
from .prompts import qa_messages


def evaluate(model, cfg, out_path=None):
    rows = load_eval(cfg)
    llm = build_llm(model, tp=1, max_model_len=cfg["eval"]["max_tokens"] + 1024)
    msgs = [qa_messages(r["question"]) for r in rows]
    outs = chat_many(llm, msgs, n=1, temperature=cfg["eval"]["temperature"],
                     top_p=1.0, max_tokens=cfg["eval"]["max_tokens"])
    n_ok = 0
    results = []
    for r, comps in zip(rows, outs):
        pred = extract_answer(comps[0].text)
        ok = score_answer(pred, r["gold"])
        n_ok += int(ok)
        results.append({**r, "pred": pred, "correct": ok, "raw": comps[0].text})
    acc = n_ok / max(len(rows), 1)
    print(f"[eval] {model}\n[eval] accuracy: {acc:.4f} ({n_ok}/{len(rows)})")
    by_level = defaultdict(lambda: [0, 0])
    for x in results:
        lv = x.get("level") or "-"
        by_level[lv][0] += int(x["correct"])
        by_level[lv][1] += 1
    if len(by_level) > 1:
        for lv in sorted(by_level):
            c, t = by_level[lv]
            print(f"[eval]   level {lv}: {c / t:.4f} ({c}/{t})")
    if out_path:
        write_jsonl(out_path, results)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(args.model or latest_student(cfg), cfg, args.out)


if __name__ == "__main__":
    main()
