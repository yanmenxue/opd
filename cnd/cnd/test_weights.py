"""weights.py 单元测试（纯 python，无 torch 依赖）：python -m cnd.test_weights"""
from .weights import cnd_step_weights, omega_advantage


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_cnd_weights():
    nec = [1.0, 0.0, None]
    spans = [(0, 2), (2, 4), (4, 6)]
    omega, st = cnd_step_weights(nec, spans, alpha=1.0)
    assert st["kept"] == 2
    assert omega[4] == 0.0 and omega[5] == 0.0      # None 步不受监督
    assert approx(sum(omega), 1.0)
    assert omega[0] > omega[2]                       # N 高 → 权重高
    omega2, st2 = cnd_step_weights([None] * 3, spans)
    assert st2["kept"] == 0 and not any(omega2)


def test_cnd_uniform_necessity():
    # 全部 N 相同 → softmax 均匀 → ω = 1/K 均摊
    nec = [0.5, 0.5]
    omega, st = cnd_step_weights(nec, [(0, 2), (2, 4)])
    assert approx(sum(omega), 1.0) and approx(omega[0], 0.25)


def test_omega_advantage_basic():
    rkl = [1.0, 3.0, 0.5, 0.5]
    omega = [0.5, 0.5, 0.0, 0.0]
    adv, mask = omega_advantage(rkl, omega, w_kl=2.0, adv_norm=False)
    assert mask == [True, True, False, False]
    assert adv[0] == -1.0 and adv[1] == -3.0


def test_omega_advantage_post_factors():
    rkl = [1.0, 3.0]
    omega = [0.5, 0.5]
    adv, _ = omega_advantage(rkl, omega, w_kl=1.0, adv_norm=False,
                             correct=True, correct_scale=0.5)
    assert approx(adv[0], -0.25)
    adv2, _ = omega_advantage(rkl, omega, adv_norm=False, acc_adv=2.0, lambda_acc=0.5)
    assert approx(adv2[0], -0.5 + 1.0)


def test_omega_advantage_norm_on_selected():
    rkl = [1.0, 3.0, 0.5, 0.5]
    omega = [0.5, 0.5, 0.0, 0.0]
    adv, mask = omega_advantage(rkl, omega, adv_norm=True)
    sel = [a for a, m in zip(adv, mask) if m]
    assert abs(sum(sel) / len(sel)) < 1e-5   # 选中集上标准化


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
