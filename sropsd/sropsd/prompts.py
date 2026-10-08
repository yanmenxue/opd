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
    """教师带参考答案提示的视图（SR-OPSD 自蒸馏设定 π_T(·|x, y*, ·)）。"""
    return [{
        "role": "user",
        "content": (
            "Solve the following math problem step by step, then give the final "
            "answer inside \\boxed{}.\n\n" + question.strip()
            + f"\n\n[Reference answer for internal calibration: {gold}]"
        ),
    }]
