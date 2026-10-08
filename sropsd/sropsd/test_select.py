"""select.py 单元测试（纯 python，无 torch 依赖）：python -m sropsd.test_select"""
from .select import reliability, retire_step, select_mask


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_none():
    mask, st = select_mask([1.0, 2.0], [0.0, 0.0], mode="none")
    assert mask == [True, True] and st["frac"] == 1.0 and st["kl_coverage"] == 1.0


def test_usefulness_topk():
    rkl = [0.0, 5.0, 0.0, 3.0]
    mask, st = select_mask(rkl, [0.0] * 4, mode="usefulness", budget=0.5, min_tokens=1)
    assert mask == [False, True, False, True]
    assert approx(st["frac"], 0.5) and approx(st["kl_coverage"], 1.0)


def test_dual_prefers_low_entropy():
    rkl = [1.0, 1.0]
    ent = [3.0, 0.0]
    m_u, _ = select_mask(rkl, ent, mode="usefulness", budget=0.5, min_tokens=1)
    assert m_u == [True, False]  # 平票取先
    m_d, _ = select_mask(rkl, ent, mode="dual", budget=0.5, min_tokens=1)
    assert m_d == [False, True]  # 双轴转向低熵（梯度可靠）位置
    assert approx(reliability(3.0), 0.25)


def test_step_retire():
    # 步1（token0-1）：低分歧+低不确定 → 退役；步2：保留
    rkl = [0.001, 0.001, 1.0, 1.0]
    ent = [0.0, 0.0, 2.0, 2.0]
    mask, st = select_mask(rkl, ent, [(0, 2), (2, 4)], mode="step_retire",
                           budget=1.0, tau_align=0.01, tau_unc=0.5, min_tokens=1)
    assert st["retired"] == 1
    assert mask == [False, False, True, True]
    assert st["kl_coverage"] > 0.99


def test_all_retired_drops_seq():
    mask, st = select_mask([0.001, 0.001], [0.0, 0.0], [(0, 2)], mode="step_retire",
                           budget=1.0, tau_align=0.01, tau_unc=0.5, min_tokens=1)
    assert not any(mask) and st["frac"] == 0.0


def test_dual_step_budget_relates_to_full_len():
    rkl = [0.0, 0.0, 0.0, 4.0, 1.0, 3.0]
    spans = [(0, 3), (3, 6)]
    mask, st = select_mask(rkl, [0.0] * 6, spans, mode="dual_step", budget=0.34,
                           tau_align=0.01, tau_unc=0.1, min_tokens=1)
    # 步1 退役；quota = round(0.34*6)=2 → 步2 中 rkl 最高的 2 个（4.0, 3.0）
    assert mask == [False] * 3 + [True, False, True]
    assert approx(st["kl_coverage"], 7.0 / 8.0)


def test_min_tokens_floor():
    rkl = [9.0] + [0.0] * 9
    mask, _ = select_mask(rkl, [0.0] * 10, mode="usefulness", budget=0.01, min_tokens=2)
    assert sum(mask) == 2


def test_retire_rule():
    assert retire_step(0.001, 0.0, 0.01, 0.5)
    assert not retire_step(0.5, 0.0, 0.01, 0.5)    # 分歧大：教师有可教的
    assert not retire_step(0.001, 2.0, 0.01, 0.5)  # 学生不确定：不该锁死


def test_missing_entropy_degrades_to_usefulness():
    rkl = [0.0, 5.0, 0.0, 3.0]
    m1, _ = select_mask(rkl, [0.0] * 4, mode="dual", budget=0.5, min_tokens=1)
    m2, _ = select_mask(rkl, [0.0] * 4, mode="usefulness", budget=0.5, min_tokens=1)
    assert m1 == m2  # 熵全零时双轴退化为仅有用性（兼容旧 graded.jsonl）


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
