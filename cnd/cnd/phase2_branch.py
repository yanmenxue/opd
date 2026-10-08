"""Phase 2: CND 步骤价值分布 —— 步骤分支 MC rollout。

对正确 rollout 的每个步骤结尾，从（题目 prompt + 前缀到该步末尾）续写 B 条
短 rollout 到答案，统计答案经验分布 P_k，得因果必要性：

    N_k = 1 − H_smooth(P_k) / log(B) ∈ [0,1]

N_k 高 = 从该步出发答案几乎被唯一决定（关键决策点）；N_k 低 = 多路可达
（松散步骤）。这是 spec 中"MCTS 访问分布熵"的可操作化：以步骤为节点的
分支 rollout（Math-Shepherd 式 MC 估计）；UCT 式干预搜索属扩展。
仅对正确 rollout 构建（spec）；necessity 与 phase3 的 token step_spans
按下标一一对应（同一 token_step_spans 切分）。
"""
import argparse
import math
import random

from .common import (
    build_llm, extract_answer, load_config, read_jsonl, round_dir, score_answer,
    smooth_entropy, token_step_spans, write_jsonl,
)
from .prompts import qa_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg.get("cnd", {})
    B = cc.get("B", 4)
    mt = cc.get("max_tokens", 256)
    max_steps = cc.get("max_steps", 0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["student_model"])
    rd = round_dir(cfg["work_dir"], args.round)
    rows = read_jsonl(rd / "rollouts.jsonl")
    llm = build_llm(cfg["student_model"], tp=1,
                    max_model_len=cfg["rounds"]["max_seq_len"] + mt + 64)

    # 打平为逐 (rollout, step) 分支任务
    rng = random.Random(cfg["train"]["seed"] + args.round)
    tasks, meta = [], []
    for ri, r in enumerate(rows):
        prompt_ids = tok.apply_chat_template(qa_messages(r["question"]),
                                             add_generation_prompt=True)
        for gi, resp in enumerate(r["responses"]):
            ok = bool(resp["pred"]) and score_answer(resp["pred"], r["gold"])
            if cc.get("only_correct", True) and not ok:
                continue
            spans = token_step_spans(tok, resp["token_ids"])
            if len(spans) < 2:
                continue  # 无可分步：无分支信号
            idxs = range(len(spans))
            if max_steps and len(spans) > max_steps:
                idxs = sorted(rng.sample(range(len(spans)), max_steps))
            for k in idxs:
                tasks.append(prompt_ids + list(resp["token_ids"][:spans[k][1]]))
                meta.append((ri, gi, k))

    print(f"[branch] {len(tasks)} branch rollouts (B={B})")
    if tasks:
        try:
            from vllm import TokensPrompt
            prompts = [TokensPrompt(prompt_token_ids=p) for p in tasks]
        except ImportError:
            prompts = tasks  # 旧版 vLLM 直接接受 token id 列表
        from vllm import SamplingParams
        sp = SamplingParams(n=B, temperature=cc.get("temperature", 0.8),
                            top_p=cc.get("top_p", 0.95), max_tokens=mt)
        outs = llm.generate(prompts, sp)
    else:
        outs = []

    log_b = math.log(max(B, 2))
    acc = {}
    for (ri, gi, k), comps in zip(meta, outs):
        counts = {}
        for c in comps:
            a = extract_answer(c.text)
            if a:
                counts[a] = counts.get(a, 0) + 1
        if counts:
            n = max(0.0, min(1.0, 1.0 - smooth_entropy(counts) / log_b))
        else:
            n = None  # 续写全部无答案：不进 softmax、不受监督
        acc.setdefault((ri, gi), {})[k] = n

    out = [{"qid": rows[ri]["qid"], "resp": gi,
            "necessity": {str(k): v for k, v in nd.items()}}
           for (ri, gi), nd in acc.items()]
    write_jsonl(rd / "cnd.jsonl", out)

    vals = [v for nd in acc.values() for v in nd.values() if v is not None]
    n_none = sum(1 for nd in acc.values() for v in nd.values() if v is None)
    if vals:
        print(f"[branch] round {args.round}: {len(out)} rollouts, mean N={sum(vals)/len(vals):.3f} "
              f"min={min(vals):.3f} unmeasured={n_none} -> cnd.jsonl")
    else:
        print(f"[branch] round {args.round}: no measurable steps -> cnd.jsonl")


if __name__ == "__main__":
    main()
