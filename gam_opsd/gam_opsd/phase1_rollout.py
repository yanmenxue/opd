"""Phase 1: 学生 on-policy rollout（vLLM，当前学生权重）。"""
import argparse

from .common import (
    build_llm, chat_many, extract_answer, latest_student, load_config, read_jsonl,
    round_dir, score_answer, write_jsonl,
)
from .prompts import qa_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--student_path", default=None, help="覆盖学生模型路径（默认取配置）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = cfg["rounds"]
    student = args.student_path or latest_student(cfg)

    prompts = read_jsonl(f"{cfg['work_dir']}/prompts.jsonl")
    # 无放回轮流取题：轮次 r 取第 [r*B, (r+1)*B) 块
    per_round = rc["prompts_per_round"]
    k = args.round
    chunk = prompts[(k * per_round) % len(prompts):][:per_round]
    if len(chunk) < per_round:  # 不足一轮则从头补齐（题循环）
        chunk = chunk + prompts[:per_round - len(chunk)]

    llm = build_llm(student, tp=1, max_model_len=rc["max_seq_len"] + 64)
    msgs = [qa_messages(r["question"]) for r in chunk]
    # logprobs=1: 返回包含采样 token 的 logprob
    outs = chat_many(llm, msgs, n=rc["G"], temperature=rc["temperature"],
                     top_p=rc["top_p"], max_tokens=rc["max_new_tokens"], logprobs=1)

    rows = []
    for r, comps in zip(chunk, outs):
        resps = []
        for c in comps:
            chosen_lp = []
            if c.logprobs:
                for lp_dict in c.logprobs:
                    lp = lp_dict.get(c.token_ids[len(chosen_lp)])
                    chosen_lp.append(lp.logprob if lp is not None else None)
            resps.append({
                "text": c.text,
                "token_ids": list(c.token_ids),
                "chosen_lp": chosen_lp,
                "pred": extract_answer(c.text),
            })
        rows.append({**r, "responses": resps})
    out_path = round_dir(cfg["work_dir"], k) / "rollouts.jsonl"
    write_jsonl(out_path, rows)
    acc = sum(any(score_ok(rr, r["gold"]) for rr in r["responses"]) for r in rows) / max(len(rows), 1)
    print(f"[rollout] round {k}: {len(rows)} prompts -> {out_path} (pass@{rc['G']}={acc:.3f})")


def score_ok(rr, gold):
    from .common import score_answer
    return bool(rr.get("pred")) and score_answer(rr["pred"], gold)


if __name__ == "__main__":
    main()
