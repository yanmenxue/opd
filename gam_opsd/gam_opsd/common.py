"""公共工具：配置加载、IO、答案抽取、步骤切分、判分、vLLM。"""
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


# ---------- 配置 ----------
@dataclass
class Cfg:
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)


def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        return Cfg(yaml.safe_load(f))


# ---------- IO ----------
def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    """追加一行（读-改-写；轮级日志需跨 round 累积，write_jsonl 会覆盖）。"""
    path = Path(path)
    rows = read_jsonl(path) if path.exists() else []
    rows.append(row)
    write_jsonl(path, rows)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def round_dir(work_dir: str, k: int) -> Path:
    d = Path(work_dir) / f"round_{k:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_student(cfg):
    """当前学生权重路径：训练过则用 ckpt_dir，否则用初始 student_model。

    checkpoint 与初始模型路径分离，避免同 backbone 自蒸馏时教师
    （从 teacher_model 加载）被学生的 checkpoint 静默遮蔽。
    """
    ckpt = Path(cfg["train"].get("ckpt_dir", "./ckpt"))
    if (ckpt / "model.safetensors").exists():
        return str(ckpt)
    return cfg["student_model"]


# ---------- 答案抽取与归一化 ----------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED_HEAD = "\\boxed{"


def last_boxed(text: str) -> str:
    """取最后一个 \\boxed{...} 的内容；支持嵌套花括号（\\frac{1}{2} 等，
    简单正则 [^{}]+ 匹配不到这类答案）。"""
    start = text.rfind(_BOXED_HEAD)
    if start == -1:
        return ""
    i, depth, out = start + len(_BOXED_HEAD), 1, []
    while i < len(text):
        c = text[i]
        if c == "\\":  # 跳过转义（\{ \} 是字面花括号，不计深度）
            out.append(text[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return ""  # 括号不闭合（截断输出），交给调用方回退


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract_answer(text: str) -> str:
    """优先 \\boxed{}，否则取最后一个数字。"""
    if not text:
        return ""
    m = last_boxed(text)
    if m:
        return normalize_answer(m)
    nums = _NUM.findall(text)
    return normalize_answer(nums[-1]) if nums else ""


# ---------- 推理步骤切分 ----------
_STEP_MARK = re.compile(r"\n\s*(?:\d+[.、)]|Step\s*\d+|第\s*\d+\s*步)")


def split_steps(rationale: str):
    """按空行/编号标记切分推理步骤；失败则按单换行粗切。"""
    if not rationale:
        return []
    parts = _STEP_MARK.split("\n" + rationale.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in rationale.split("\n") if p.strip()]
    return parts


def token_step_spans(tok, token_ids, window=64, min_step_tokens=4):
    """响应 token 序列的步骤 span [(s,e),...]（token 下标，左闭右开）。

    优先解码成文本、用 fast tokenizer 的 offset mapping 把步骤标记对齐回
    token 边界；重编码不一致（如尾部 eos）或 tokenizer 不支持 offsets 时
    退化为固定窗口。返回 [] 表示无可分步，调用方按单步 (0,n) 处理。
    """
    n = len(token_ids)
    if n == 0:
        return []
    ids = list(token_ids)
    try:
        text = tok.decode(ids, skip_special_tokens=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != ids:
            while ids and ids[-1] in (tok.eos_token_id, tok.pad_token_id):
                ids.pop()
            if not ids:
                raise ValueError("empty after stripping specials")
            text = tok.decode(ids, skip_special_tokens=False)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            if list(enc["input_ids"]) != ids:
                raise ValueError("re-encode mismatch")
        offsets = enc["offset_mapping"]
    except Exception:
        return [(s, min(s + window, n)) for s in range(0, n, window)]

    starts = [0]
    for i, (cs, ce) in enumerate(offsets):
        if ce <= cs or cs == 0 or text[cs - 1] != "\n":
            continue
        if _STEP_MARK.match(text, cs - 1) or (cs < len(text) and text[cs] == "\n"):
            if i > starts[-1]:
                starts.append(i)
    if len(starts) < 2:
        return []
    bounds = starts + [n]
    spans = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    merged = []
    for s, e in spans:  # 过短步并入前一步
        if merged and e - s < min_step_tokens:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ---------- 数学判分 ----------
def score_answer(pred: str, gold: str) -> bool:
    """math_verify 优先，失败回退字符串比较。"""
    if not pred:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return normalize_answer(pred) == normalize_answer(gold)


# ---------- chat 模板 ----------
def chat_template_ids(tok, conversation, enable_thinking=None):
    """apply_chat_template 渲染成字符串后手动编码为 ids；不依赖各 transformers
    版本的 tokenize 默认值与返回类型（v5 起默认 False 返回 str）。

    enable_thinking=None 时不传该参数（沿用模板默认；实测 Qwen3.5 的 {} 与
    enable_thinking=True 渲染一致，即默认「思考开」）。phase1 走 vLLM 的 chat()
    用它自己的模板默认值，两边须保持同一个值，否则师生 prompt 不一致。"""
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tok.apply_chat_template(conversation, add_generation_prompt=True,
                                   tokenize=False, **kw)
    return tok(text, add_special_tokens=False)["input_ids"]


# ---------- vLLM ----------
def build_serving_dir(cfg):
    """vLLM 可服务的完整模型目录（文本骨干 = 训练后学生权重）。

    基座是 VL 模型（Qwen3_5ForConditionalGeneration），ckpt 只存文本骨干权重，
    而服务器 vLLM 未注册 Qwen3.5 纯文本架构（Qwen3Next 实现为 MoE 专用，
    装不下稠密权重）——故整体复制原始模型目录（config/分词器/视觉处理器），
    将文本权重张量替换为学生 ckpt；vLLM 按原始 VL 架构加载，视觉塔闲置。
    无训练 ckpt 时直接返回原始路径。"""
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(cfg["student_model"])
    ckpt_w = Path(cfg["train"].get("ckpt_dir", "./ckpt")) / "model.safetensors"
    if not ckpt_w.exists():
        return str(src)
    out = Path(cfg["work_dir"]) / "serving_student"
    sig = f"v3:{ckpt_w}:{ckpt_w.stat().st_mtime_ns}"  # v3: 补非权重文件复制 + 顶层键对齐
    marker = out / ".built_for"
    if marker.exists() and marker.read_text().strip() == sig:
        return str(out)  # ckpt 未变，直接复用
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():  # config/tokenizer/processor 必须一起带上，否则 vLLM 起不来
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    files = sorted(src.glob("*.safetensors"))
    orig_keys = set()
    for f in files:
        with safe_open(f, framework="pt") as sf:
            orig_keys.update(sf.keys())
    stud = safe_open(ckpt_w, framework="pt")
    stud_keys = set(stud.keys())
    # ckpt 是文本子模块命名（model.* / lm_head.*），原始 VL 权重带
    # language_model. 前缀——以原始键为准做前缀对齐，只替换能对上的文本张量
    prefix = ""
    for k in sorted(orig_keys):
        if k.endswith("model.embed_tokens.weight"):
            prefix = k[:-len("model.embed_tokens.weight")]
            break
    repl = {prefix + k: k for k in stud_keys if prefix + k in orig_keys}
    repl.update({k: k for k in stud_keys if k in orig_keys})  # 顶层键（lm_head 等）同名对齐
    if not repl:
        raise RuntimeError("ckpt 键与原始权重对齐失败（前缀不匹配），不重建 serving 目录")
    miss = [k for k in stud_keys if k not in repl]
    if miss:
        print(f"[serving] 警告：{len(miss)} 个学生张量未替换，服务端仍用基座权重 {miss}")
    print(f"[serving] 前缀 '{prefix}'：替换张量 {len(repl)}/{len(stud_keys)}，"
          f"其余 {len(orig_keys) - len(set(repl.values()))} 个保持原样")
    merged = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                merged[k] = (stud.get_tensor(repl[k]).to(torch.bfloat16)
                             if k in repl else sf.get_tensor(k))
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    marker.write_text(sig)
    print(f"[serving] 文本权重已替换为学生 ckpt -> {out}")
    return str(out)


def build_llm(model: str, tp: int, max_model_len: int = 4096, mem_util: float = 0.92):
    from vllm import LLM
    return LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        gpu_memory_utilization=mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )


def chat_many(llm, messages_list, n, temperature, top_p, max_tokens, seed=42, logprobs=None):
    """批量 chat 采样。返回 list[list[CompletionOutput]]。"""
    from vllm import SamplingParams
    sp = SamplingParams(
        n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        seed=seed, logprobs=logprobs,
    )
    outs = llm.chat(messages_list, sp, use_tqdm=True)
    return [[c for c in o.outputs] for o in outs]
