"""GAM-OPSD 粒度路由核心（纯 python，可单测）。

问题：token 级信用分配在"弥散信用"状态下有统计效率的二次惩罚——当一个
推理操作的回报效应分散在步骤内 m 个 token 上（每 token 效应 Θ(|A_step|/m)）
时，区分有益/有害 token 所需 MC 预算 Θ(m²)，而步骤级只需常数预算。

信用弥散度：局部窗口内师生对数比 a 的归一化方差

    a_k = log π_T(y_k | x, y*, ŷ_<k) − log π_S(y_k | ŷ_<k)
    Disp_t = Var_{k∈W_t}[a_k] / (Var[a] + ε)      W_t = [t−w, t+w]（截断到序列）

Disp 高 = 分歧均匀弥散在窗口内（弥散信用）→ token 级信号不可靠 → 聚合到
步骤级；Disp 低 = 分歧集中在少数位置（集中信用）→ 保持 token 级。

粒度路由：granularity_t = step if Disp_t > τ else token（τ 默认 0.5）。
步骤级损失以步骤为单位，故按步骤聚合判定：步骤 k 弥散 ⟺ 步内 mean Disp > τ。

步骤级效应求和而非平均：A_step = Σ_{t∈S_k} a_t——求和让信号从 |A_step|/m
恢复到 |A_step|，噪声 std 只增长 √m 倍，SNR 改善 √m。

训练形态（离线打分 + 单 epoch REINFORCE 代理，同 sropsd/cnd 架构）：

    L_GAM = Σ_{t: token 级} ℓ_t^token + λ_step Σ_{k: 弥散步} ℓ_k^step

  per-token 形式：A_t = w_kl·a_t（token 级位置）；A_t = λ_step·w_kl·A_step/|S_k|
  （弥散步内所有 token 平摊步骤优势，等价于 ℓ_k^step = −A_step·mean_{t∈S_k} log π）。
  a_t 方向：>0 = 教师比学生更确信采样 token → 提升之；E_{y~π_S}[a_t] = −KL_t，
  即单样本前向 KL 的 REINFORCE 代理。adv_norm 只在 token 分支内 z-norm
  （a 的符号本身携带方向，全量 z-norm 会抹掉均值中的蒸馏压力）。
"""
import math


def dispersion(a, window=16, eps=1e-6):
    """逐 token 信用弥散度 Disp_t = Var_{W_t}[a] / (Var[a] + eps)。

    两级方差均为总体方差（1/n）；窗口在序列边界截断。前缀和 O(n) 实现。
    a 全相等时全局方差为 0 → Disp ≡ 0（无弥散信息，全部路由 token 级）。
    """
    n = len(a)
    if n == 0:
        return []
    pre = [0.0] * (n + 1)
    pre2 = [0.0] * (n + 1)
    for i, x in enumerate(a):
        pre[i + 1] = pre[i] + x
        pre2[i + 1] = pre2[i] + x * x
    mu = pre[n] / n
    var = pre2[n] / n - mu * mu
    out = []
    for t in range(n):
        lo, hi = max(0, t - window), min(n, t + window + 1)
        m = (pre[hi] - pre[lo]) / (hi - lo)
        v = (pre2[hi] - pre2[lo]) / (hi - lo) - m * m
        out.append(v / (var + eps))
    return out


def route_steps(disp, spans, tau=0.5, mode="dynamic"):
    """每个步骤的粒度标签。返回 is_step: list[bool]（True=步骤级/弥散步）。

    mode: dynamic 按 Disp 路由 | token 全 token 级（标准 OPSD 臂） |
          step 全步骤级（全步骤级 OPSD 臂）——固定粒度消融共用此开关。
    """
    if mode == "step":
        return [True] * len(spans)
    if mode == "token":
        return [False] * len(spans)
    if mode != "dynamic":
        raise ValueError(f"unknown route.mode: {mode}")
    return [sum(disp[s:e]) / (e - s) > tau for s, e in spans]


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def route_diag(a, spans, disp, is_step):
    """单响应路由诊断（spec 关键诊断：弥散度分布 + 弥散度×步长相关）。

    预期发现：信用弥散度与步骤内 token 数正相关（corr_disp_steplen > 0）。
    """
    n = len(a)
    if n == 0:
        return {}
    ds = sorted(disp)
    med = ds[n // 2] if n % 2 else 0.5 * (ds[n // 2 - 1] + ds[n // 2])
    steplen = []
    for s, e in spans:
        steplen.extend([e - s] * (e - s))
    return {
        "disp_mean": sum(disp) / n,
        "disp_median": med,
        "disp_p25": ds[n // 4],
        "disp_p75": ds[(3 * n) // 4],
        "frac_step_tokens": sum(e - s for (s, e), st in zip(spans, is_step) if st) / n,
        "n_steps": len(spans),
        "frac_steps_routed": sum(is_step) / max(len(spans), 1),
        "corr_disp_steplen": _pearson(disp, steplen),
    }


def gam_advantage(a, spans, mode="dynamic", tau=0.5, window=16, eps=1e-6,
                  w_kl=1.0, lambda_step=1.0, adv_norm=False,
                  correct=False, correct_scale=1.0, acc_adv=None, lambda_acc=0.0):
    """L_GAM 的 per-token 优势。返回 (adv, mask, info)。

    - adv 与 a 等长；mask 恒全 True：两个分支覆盖全部位置，粒度只改优势
      聚合层级、不改监督区域（mask 保留以兼容训练器接口）；
    - token 级位置 A_t = w_kl·a_t（adv_norm=true 时先在 token 分支内 z-norm）；
    - 弥散步 A_step = w_kl·Σ_{t∈S_k} a_t（求和而非平均），步内平摊
      A_t = λ_step·A_step/|S_k|；
    - 常因子（correct_scale）与 acc 项在最后施加，加在全向量上。
    info 为 route_diag 的路由诊断（mode≠dynamic 时弥散度字段为 None）。
    """
    n = len(a)
    if n == 0:
        return [], [], {}
    spans = spans or [(0, n)]
    disp = dispersion(a, window, eps) if mode == "dynamic" else None
    is_step = route_steps(disp, spans, tau, mode)

    adv = [0.0] * n
    tok_pos = [t for (s, e), st in zip(spans, is_step) for t in range(s, e) if not st]
    a_tok = {t: a[t] for t in tok_pos}
    if adv_norm and tok_pos:
        mu = sum(a_tok.values()) / len(tok_pos)
        sd = (sum((x - mu) ** 2 for x in a_tok.values()) / len(tok_pos)) ** 0.5
        a_tok = {t: (x - mu) / (sd + 1e-6) for t, x in a_tok.items()}
    for t, x in a_tok.items():
        adv[t] = w_kl * x
    for (s, e), st in zip(spans, is_step):
        if st:
            a_step = sum(a[s:e])                     # 求和而非平均
            v = lambda_step * w_kl * a_step / (e - s)
            for t in range(s, e):
                adv[t] = v

    if correct and correct_scale != 1.0:
        adv = [correct_scale * x for x in adv]
    if lambda_acc and acc_adv is not None:
        acc = lambda_acc * float(acc_adv)
        adv = [x + acc for x in adv]

    info = route_diag(a, spans, disp, is_step) if disp is not None else {
        "disp_mean": None, "disp_median": None, "disp_p25": None, "disp_p75": None,
        "frac_step_tokens": sum(e - s for (s, e), st in zip(spans, is_step) if st) / n,
        "n_steps": len(spans),
        "frac_steps_routed": sum(is_step) / max(len(spans), 1),
        "corr_disp_steplen": None,
    }
    return adv, [True] * n, info
