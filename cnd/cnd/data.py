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
    """MATH train。hendrycks 原库是 script 数据集（datasets>=3.0 已移除支持），
    失败回退 parquet 版 EleutherAI/hendrycks_math（同字段 problem/solution）。"""
    import re

    from datasets import load_dataset
    try:
        ds = load_dataset("hendrycks/competition_math", split="train",
                          trust_remote_code=True)
    except Exception:
        ds = load_dataset("EleutherAI/hendrycks_math", split="train")
    rows = []
    for i, ex in enumerate(ds):
        # gold 在 solution 的 \boxed{} 中
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


def _load_aime24():
    """AIME 2024（30 题，OOD-难）。答案为 0-999 整数；HF id 失效时用 jsonl:<path> 兜底。"""
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [{"qid": f"aime24-{i}", "question": ex["Problem"], "gold": str(ex["Answer"])}
            for i, ex in enumerate(ds)]


def _load_amc23():
    """AMC 2023（40 题，OOD-难）。HF id 失效时用 jsonl:<path> 兜底。"""
    from datasets import load_dataset
    ds = load_dataset("zwhe99/amc23", split="test")
    return [{"qid": f"amc23-{i}", "question": ex["question"], "gold": str(ex["answer"])}
            for i, ex in enumerate(ds)]


def _load_aime25():
    """AIME 2025（30 题，OOD-难）。HF id 未验证，依次尝试，失效用 jsonl:<path> 兜底。"""
    from datasets import load_dataset
    last = None
    for repo, qf, af in [("Maxwell-Jia/AIME_2025", "Problem", "Answer"),
                         ("MathArena/aime_2025", "problem", "answer")]:
        try:
            ds = load_dataset(repo, split="train")
            return [{"qid": f"aime25-{i}", "question": ex[qf], "gold": str(ex[af])}
                    for i, ex in enumerate(ds)]
        except Exception as e:  # noqa: BLE001 — 逐个候选尝试
            last = e
    raise RuntimeError(f"aime25 loader failed (candidates unreachable: {last}); "
                       "用 data_eval: jsonl:<path> 兜底")


def load_eval(cfg):
    src = cfg["data_eval"]
    if src == "gsm8k_test":
        rows = _load_gsm8k("test")
    elif src == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"qid": f"m500-{i}", "question": ex["problem"], "gold": ex["answer"],
                 "level": ex.get("level", "")}
                for i, ex in enumerate(ds)]
    elif src == "aime24":
        rows = _load_aime24()
    elif src == "aime25":
        rows = _load_aime25()
    elif src == "amc23":
        rows = _load_amc23()
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
