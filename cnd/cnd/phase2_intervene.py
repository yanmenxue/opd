"""Phase 2: Causal-Credit 反事实干预 —— 步骤矛盾替换 + 后续重生成。

对正确 rollout（长度 > min_rollout_tokens）的每个步骤 k：
1. 用学生模型把 s_k 改写为语义矛盾、同粒度的 s_k'（V 个变体，干预类型可配：
   negate 逻辑反转 / factual 事实反转 / premise 前提否定）；
2. 从干预点（prompt + 步骤 1..k-1 + s_k'）重新生成后续推理与答案 a*；
3. 因果敏感度 φ_k = mean_V [1 − Sim(a, a*)]，Sim = math_verify 等价（精确匹配）。

φ 高 = 因果必要（改它答案变）；φ 低 = 因果解耦（装饰性步骤）。变体全部无答案
→ φ=None（不进 softmax、不受监督）。
与 CND 分支 MC（phase2_branch，现为消融臂）的区别：分支 MC 测"该状态多路
可达性"（中性续写），本相位测"答案对该步内容的因果敏感度"（对抗替换）。
两者共享 softmax(α·per-step 标量) 的权重形式。
"""
import argparse
import random

from .common import (
    build_llm, chat_many, extract_answer, load_config, read_jsonl, round_dir,
    score_answer, token_step_spans, write_jsonl,
)
from .prompts import cc_intervention_messages, qa_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg.get("cc", {})
    V = cc.get("variants", 2)
    kind = cc.get("intervention", "negate")
    mt = cc.get("max_tokens", 256)
    min_toks = cc.get("min_rollout_tokens", 300)
    max_steps = cc.get("max_steps", 0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["student_model"])
    rd = round_dir(cfg["work_dir"], args.round)
    rows = read_jsonl(rd / "rollouts.jsonl")
    llm = build_llm(cfg["student_model"], tp=1,
                    max_model_len=cfg["rounds"]["max_seq_len"] + mt + 64)

    # 收集干预点：(ri, gi, k, step_text, prefix_ids)；prefix = prompt + 步骤 1..k-1
    rng = random.Random(cfg["train"]["seed"] + args.round)
    steps, n_skip_len, n_skip_span = [], 0, 0
    for ri, r in enumerate(rows):
        prompt_ids = tok.apply_chat_template(qa_messages(r["question"]),
                                             add_generation_prompt=True)
        for gi, resp in enumerate(r["responses"]):
            ok_i = bool(resp["pred"]) and score_answer(resp["pred"], r["gold"])
            if not ok_i:  # spec：只对正确 rollout 施加干预
                continue
            if len(resp["token_ids"]) < min_toks:
                n_skip_len += 1
                continue
            spans = token_step_spans(tok, resp["token_ids"])
            if len(spans) < 2:
                n_skip_span += 1
                continue
            idxs = range(len(spans))
            if max_steps and len(spans) > max_steps:
                idxs = sorted(rng.sample(range(len(spans)), max_steps))
            for k in idxs:
                step_text = tok.decode(resp["token_ids"][spans[k][0]:spans[k][1]],
                                       skip_special_tokens=True).strip()
                if not step_text:
                    continue
                prefix = prompt_ids + list(resp["token_ids"][:spans[k][0]])
                steps.append((ri, gi, k, r["question"], step_text, prefix))

    print(f"[cc] {len(steps)} intervention points "
          f"(skipped: short-rollout {n_skip_len}, no-steps {n_skip_span}), "
          f"kind={kind}, V={V}")
    if not steps:
        write_jsonl(rd / "cc.jsonl", [])
        print(f"[cc] round {args.round}: nothing to intervene -> cc.jsonl")
        return

    # Round 1：矛盾改写（chat，n=V）
    msgs = [cc_intervention_messages(q, st, kind) for (_, _, _, q, st, _) in steps]
    rewrites = chat_many(llm, msgs, n=V, temperature=cc.get("temperature", 0.8),
                         top_p=cc.get("top_p", 0.95),
                         max_tokens=cc.get("rewrite_max_tokens", 128))

    # Round 2：从干预点重生成到答案
    cap = cc.get("rewrite_max_tokens", 128)
    cont_tasks, cont_meta, n_variant = [], [], 0
    for (ri, gi, k, _, _, prefix), comps in zip(steps, rewrites):
        for c in comps:
            vtext = c.text.strip()
            if not vtext:
                continue
            n_variant += 1
            vid = tok(vtext, add_special_tokens=False)["input_ids"][:cap]
            cont_tasks.append(prefix + vid)
            cont_meta.append((ri, gi, k))

    from vllm import SamplingParams
    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=p) for p in cont_tasks]
    except ImportError:
        prompts = cont_tasks
    outs = llm.generate(prompts, SamplingParams(
        n=1, temperature=cc.get("temperature", 0.8),
        top_p=cc.get("top_p", 0.95), max_tokens=mt)) if cont_tasks else []

    # φ_k = mean(1 − Sim(a, a*))；无答案变体不计入；全无答案 → None
    flips, measured = {}, set()
    for (ri, gi, k), comp in zip(cont_meta, outs):
        measured.add((ri, gi, k))
        a_star = extract_answer(comp[0].text)
        if not a_star:
            continue
        a_orig = rows[ri]["responses"][gi]["pred"]
        sim = 1.0 if score_answer(a_orig, a_star) else 0.0
        flips.setdefault((ri, gi, k), []).append(1.0 - sim)

    acc = {}
    for (ri, gi, k) in sorted(measured):
        vals = flips.get((ri, gi, k))
        acc.setdefault((ri, gi), {})[k] = sum(vals) / len(vals) if vals else None

    out = [{"qid": rows[ri]["qid"], "resp": gi,
            "phi": {str(k): v for k, v in nd.items()}}
           for (ri, gi), nd in acc.items()]
    write_jsonl(rd / "cc.jsonl", out)

    vals = [v for nd in acc.values() for v in nd.values() if v is not None]
    n_none = sum(1 for nd in acc.values() for v in nd.values() if v is None)
    n_ans = sum(len(v) for v in flips.values())
    if vals:
        print(f"[cc] round {args.round}: {len(out)} rollouts, mean φ={sum(vals)/len(vals):.3f} "
              f"min={min(vals):.3f} | variants used={n_variant}, "
              f"answered-rate={n_ans/max(n_variant,1):.2f}, unmeasured={n_none} -> cc.jsonl")
    else:
        print(f"[cc] round {args.round}: no measurable steps -> cc.jsonl")


if __name__ == "__main__":
    main()
