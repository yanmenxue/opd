"""Prompt 构造：学生视图 / 教师 gold 提示视图。"""


def qa_messages(question: str):
    return [{
        "role": "user",
        "content": (
            "Solve the following math problem step by step, then give the final "
            "answer inside \\boxed{}.\n\n" + question.strip()
        ),
    }]


def qa_messages_gold(question: str, gold: str):
    """教师带参考答案提示的视图（蒸馏目标 π_T(·|x, y*, ·)）。"""
    return [{
        "role": "user",
        "content": (
            "Solve the following math problem step by step, then give the final "
            "answer inside \\boxed{}.\n\n" + question.strip()
            + f"\n\n[Reference answer for internal calibration: {gold}]"
        ),
    }]


# ---------- Causal-Credit 反事实干预模板 ----------
# 三类干预（消融轴）：逻辑反转 / 事实反转 / 前提否定。
# 共同要求：同语义粒度（仍是单步推导，不是随机替换）、同格式长度。
_CC_INTERVENTION = {
    "negate": (
        "Below is one step of a math solution.\n\n"
        "[Problem]\n{q}\n\n[Step]\n{step}\n\n"
        "Rewrite this single step so it asserts the OPPOSITE mathematical "
        "relationship (flip a key sign, comparison, or conclusion) while keeping "
        "the same format, length, and role as one solution step. "
        "Output ONLY the rewritten step, no explanation."
    ),
    "factual": (
        "Below is one step of a math solution.\n\n"
        "[Problem]\n{q}\n\n[Step]\n{step}\n\n"
        "Rewrite this single step with one key numeric value changed to a "
        "different plausible value, keeping the same format, length, and role "
        "as one solution step. Output ONLY the rewritten step, no explanation."
    ),
    "premise": (
        "Below is one step of a math solution.\n\n"
        "[Problem]\n{q}\n\n[Step]\n{step}\n\n"
        "Rewrite this single step so it relies on the NEGATION of the assumption "
        "or fact it currently uses, keeping the same format, length, and role as "
        "one solution step. Output ONLY the rewritten step, no explanation."
    ),
}


def cc_intervention_messages(question: str, step: str, kind: str = "negate"):
    """生成步骤 s_k 的矛盾版本 s_k' 的改写 prompt（Causal-Credit 干预）。"""
    tpl = _CC_INTERVENTION.get(kind, _CC_INTERVENTION["negate"])
    return [{"role": "user", "content": tpl.format(q=question.strip(), step=step.strip())}]
