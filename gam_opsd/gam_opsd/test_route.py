"""route.py 单元测试（纯 python，无 torch 依赖）：python -m gam_opsd.test_route"""
from .route import dispersion, gam_advantage, route_steps


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_dispersion_constant_zero():
    disp = dispersion([2.0, 2.0, 2.0, 2.0], window=1)
    assert all(d < 1e-9 for d in disp)  # 全局方差 0 → 无弥散信息


def test_dispersion_spike_localizes():
    # 孤立尖峰在 t=3；远离尖峰的平坦窗口 Disp≈0，尖峰处 >1（窗口方差可超全局方差）
    a = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    disp = dispersion(a, window=2)
    assert disp[3] > 1.0
    assert disp[0] < 0.01 and disp[6] < 0.01


def test_dispersion_empty():
    assert dispersion([]) == []


def test_route_modes():
    disp = [0.0, 0.9, 0.9]
    spans = [(0, 1), (1, 3)]
    assert route_steps(disp, spans, tau=0.5, mode="token") == [False, False]
    assert route_steps(disp, spans, tau=0.5, mode="step") == [True, True]
    # 步内 mean Disp：步1=0（token 级），步2=0.9（步骤级）
    assert route_steps(disp, spans, tau=0.5, mode="dynamic") == [False, True]
    try:
        route_steps(disp, spans, mode="bad")
        assert False
    except ValueError:
        pass


def test_adv_token_mode_is_standard_opsd():
    a = [1.0, -2.0, 0.5, 3.0]
    adv, mask, _ = gam_advantage(a, None, mode="token", w_kl=2.0)
    assert mask == [True] * 4
    assert all(approx(x * 2.0, y) for x, y in zip(a, adv))


def test_adv_step_mode_sum_not_mean():
    # A_step = Σa（求和）；步内平摊 A_t = A_step/|S_k|
    adv, _, _ = gam_advantage([3.0, 1.0], [(0, 2)], mode="step")
    assert adv == [2.0, 2.0]
    # 同一条序列切成两步：每步各自的 Σa——聚合层级改变优势结构
    adv2, _, _ = gam_advantage([3.0, 1.0], [(0, 1), (1, 2)], mode="step")
    assert adv2 == [3.0, 1.0]


def test_adv_dynamic_mixed_branches():
    # 步1 全平坦（Disp≈0 → token 级），步2 含远离边界的尖峰（Disp 高 → 步骤级）
    a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    spans = [(0, 4), (4, 10)]
    adv, mask, info = gam_advantage(a, spans, mode="dynamic", tau=0.5, window=2)
    assert mask == [True] * 10
    assert adv[:4] == [0.0, 0.0, 0.0, 0.0]      # token 级且 a=0
    assert approx(adv[4], 10.0 / 6.0)           # 步2: Σa=10, /6 平摊
    assert approx(info["frac_step_tokens"], 0.6)


def test_lambda_step_scales_step_branch_only():
    a = [1.0, 0.0]
    adv_tok, _, _ = gam_advantage(a, [(0, 2)], mode="token", lambda_step=5.0)
    adv_step, _, _ = gam_advantage(a, [(0, 2)], mode="step", lambda_step=5.0)
    assert adv_tok == [1.0, 0.0]                # token 分支不受 λ_step 影响
    assert adv_step == [2.5, 2.5]               # 步骤分支 ×5


def test_adv_norm_token_branch_only():
    a = [1.0, 2.0, 3.0, 4.0]
    adv, _, _ = gam_advantage(a, [(0, 4)], mode="token", adv_norm=True)
    assert approx(sum(adv), 0.0, tol=1e-5)      # token 分支 z-norm 后均值 0
    adv2, _, _ = gam_advantage(a, [(0, 4)], mode="step", adv_norm=True)
    assert adv2 == [2.5, 2.5, 2.5, 2.5]         # 步骤分支不受 norm 影响


def test_correct_scale_and_acc():
    a = [1.0, 1.0]
    adv, _, _ = gam_advantage(a, None, mode="token", correct=True, correct_scale=0.5)
    assert adv == [0.5, 0.5]
    adv2, _, _ = gam_advantage(a, None, mode="token", acc_adv=2.0, lambda_acc=0.5)
    assert adv2 == [2.0, 2.0]                   # acc 加在全向量
    adv3, _, _ = gam_advantage(a, [(0, 2)], mode="step", acc_adv=2.0, lambda_acc=0.5)
    assert adv3 == [2.0, 2.0]


def test_zero_disp_routes_token():
    # a 全相等 → Disp≡0 → dynamic 路由全 token 级（adv_norm 关闭时保留 a 原值）
    a = [0.7] * 6
    adv, _, info = gam_advantage(a, [(0, 3), (3, 6)], mode="dynamic", tau=0.5)
    assert adv == [0.7] * 6
    assert info["frac_step_tokens"] == 0.0


def test_nonzero_flat_keeps_token_values():
    # 全局有方差但每步内部平坦：token 级位置保留 a 原值
    a = [1.0, 1.0, 3.0, 3.0]
    adv, _, info = gam_advantage(a, [(0, 2), (2, 4)], mode="dynamic", tau=0.5, window=1)
    assert adv == [1.0, 1.0, 3.0, 3.0]
    assert info["frac_step_tokens"] == 0.0


def test_empty_response():
    adv, mask, info = gam_advantage([], [])
    assert adv == [] and mask == [] and info == {}


def test_diag_corr_disp_steplen_positive():
    # 长步含弥散尖峰、短步平坦 → corr(Disp, 步长) > 0（spec 预期发现）
    a = [0.0, 0.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    spans = [(0, 2), (2, 10)]
    adv, mask, info = gam_advantage(a, spans, mode="dynamic", tau=0.5, window=2)
    assert info["corr_disp_steplen"] is not None and info["corr_disp_steplen"] > 0.0
    assert info["frac_steps_routed"] == 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
