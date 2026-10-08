"""weights.py 单元测试（纯 python，无 torch 依赖）：python -m sropsd.test_weights"""
from .weights import omega_advantage, sr_step_weights


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_sr_dual_axis():
    logratio = [1.0, 1.0, 0.0, 0.0, 2.0, 2.0]
    ier = [0.9, 0.9, 0.1, 0.1, 0.5, 0.5]
    spans = [(0, 2), (2, 4), (4, 6)]
    omega, st = sr_step_weights(logratio, ier, spans, tau_u=0.0, tau_r=0.3)
    assert st["kept"] == 2          # 步2 被 τ_U 门掉（U=0）
    assert approx(sum(omega), 1.0)  # Σω = 1（softmax 归一）
    assert omega[2] == 0.0 and omega[3] == 0.0
    assert omega[4] > omega[0]      # U·R: 步3=1.0 > 步1=0.9


def test_sr_no_gates():
    logratio = [1.0, 2.0, 3.0]
    ier = [0.2, 0.5, 0.8]
    omega, st = sr_step_weights(logratio, ier, [(0, 1), (1, 2), (2, 3)])
    assert st["kept"] == 3 and approx(sum(omega), 1.0)
    assert omega[2] > omega[1] > omega[0]  # softmax(U·R) 单调


def test_sr_all_gated():
    omega, st = sr_step_weights([0.0, 0.0], [0.9, 0.9], [(0, 1), (1, 2)], tau_u=0.5)
    assert st["kept"] == 0 and not any(omega)


def test_omega_advantage_basic():
    rkl = [1.0, 3.0, 0.5, 0.5]
    omega = [0.5, 0.5, 0.0, 0.0]
    adv, mask = omega_advantage(rkl, omega, w_kl=2.0, adv_norm=False)
    assert mask == [True, True, False, False]
    assert adv[0] == -1.0 and adv[1] == -3.0


def test_omega_advantage_post_factors():
    rkl = [1.0, 3.0]
    omega = [0.5, 0.5]
    # correct_scale 在 norm 之后相乘
    adv, _ = omega_advantage(rkl, omega, w_kl=1.0, adv_norm=False,
                             correct=True, correct_scale=0.5)
    assert approx(adv[0], -0.25)
    # acc 项加在全向量
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
