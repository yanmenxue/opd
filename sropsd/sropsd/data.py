"""训练/评测数据加载与 prompts 构建。"""
import random

from .common import read_jsonl, write_jsonl


def _load_gsm8k(split="train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for i, ex in enumerate(ds):
        gold = ex["answer"].split("####")[-1].strip()
        rows.append({"qid": f"gsm8k-{split}-{i}", "question": ex["question"], "gold": gold})
    return rows


def _load_math_train():
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="train")
    rows = []
    for i, ex in enumerate(ds):
        # gold 在 solution 的 \boxed{} 中
        import re
        m = re.findall(r"\\boxed\{([^{}]+)\}", ex["solution"])
        rows.append({"qid": f"math-{i}", "question": ex["problem"],
                     "gold": m[-1] if m else ""})
    return rows


def load_train(cfg):
    src = cfg["data_train"]
    if src == "gsm8k":
        rows = _load_gsm8k("train")
    elif src == "math":
        rows = _load_math_train()
    elif src.startswith("jsonl:"):
        rows = read_jsonl(src[len("jsonl:"):])
    else:
        raise ValueError(f"unknown data_train: {src}")
    max_n = cfg.get("data", {}).get("max_questions", 0)
    if max_n and len(rows) > max_n:
        random.Random(cfg["train"]["seed"]).shuffle(rows)
        rows = rows[:max_n]
    return rows


def load_eval(cfg):
    src = cfg["data_eval"]
    if src == "gsm8k_test":
        rows = _load_gsm8k("test")
    elif src == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"qid": f"m500-{i}", "question": ex["problem"], "gold": ex["answer"]}
                for i, ex in enumerate(ds)]
    elif src.startswith("jsonl:"):
        rows = read_jsonl(src[len("jsonl:"):])
    else:
        raise ValueError(f"unknown data_eval: {src}")
    max_n = cfg["eval"].get("max_questions", 500)
    return rows[:max_n] if max_n else rows


def build_prompts(cfg):
    rows = load_train(cfg)
    write_jsonl(f"{cfg['work_dir']}/prompts.jsonl", rows)
    print(f"[data] {len(rows)} training questions -> prompts.jsonl")
