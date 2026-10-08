"""SR-OPSD 步级加权核心（纯 python，可单测）。

训练形态：spec 的直接加权 KL 损失需要在训练循环内做教师前向；本管线采用
离线打分 + 单 epoch REINFORCE 的等价代理（spec 自注"等价于用师生 log-ratio
作为步骤级 advantage 修正项"）：

    A_t = w_kl · ω_t · (−RKL_t)，其中 ω_t = w_k / |S_k|，Σ_t ω_t = 1/rollout。

直接 k3 形式（教师进训练循环）留作消融，不在本实现内。

  轴一 U_k = mean_{t∈S_k} logratio_t   师生 log-ratio（>0 = 学生比教师更自信）
  轴二 R_k = mean_{t∈S_k} IER_t        梯度信噪比（phase2 落盘）
  门控 [U_k>τ_U]·[R_k>τ_R]，w = softmax(α·U_k·R_k)
"""


def _softmax(xs):
    import math
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def sr_step_weights(logratio, ier, spans, tau_u=None, tau_r=None, alpha=1.0):
    """SR-OPSD：双轴门控 + softmax(U·R) → ω_t。

    返回 (omega, stats)。omega 长度 = spans[-1][1]（= 响应 token 数）；
    stats.kept 为门控后步数，U/R 为步级分数（日志用）。
    """
    n = spans[-1][1] if spans else 0
    if not spans:
        return [0.0] * n, {"kept": 0}
    us, rs = [], []
    for s, e in spans:
        us.append(sum(logratio[s:e]) / (e - s))
        rs.append(sum(ier[s:e]) / (e - s))
    kept = [k for k in range(len(spans))
            if (tau_u is None or us[k] > tau_u) and (tau_r is None or rs[k] > tau_r)]
    if not kept:
        return [0.0] * n, {"kept": 0, "U": us, "R": rs}
    w = _softmax([alpha * us[k] * rs[k] for k in kept])
    omega = [0.0] * n
    for wi, k in enumerate(kept):
        s, e = spans[k]
        for t in range(s, e):
            omega[t] = w[wi] / (e - s)
    return omega, {"kept": len(kept), "U": us, "R": rs,
                   "w": {k: round(x, 4) for k, x in zip(kept, w)}}


def omega_advantage(rkl, omega, w_kl=1.0, adv_norm=True,
                    correct=False, correct_scale=1.0, acc_adv=None, lambda_acc=0.0):
    """ω 加权优势：A_t = w_kl·ω_t·(−RKL_t)。

    - correct_scale 是常因子，在归一化之后施加（z-norm 会精确消掉 norm 前
      的常数因子）；
    - adv_norm 在 ω>0 的 token 上标准化（保持与 baseline 臂可比的梯度尺度）；
    - acc 项（opd_acc 臂）加在全向量上，由 mask 控制损失区。
    返回 (adv list, mask list)。
    """
    adv = [-w_kl * w * k for k, w in zip(rkl, omega)]
    mask = [w > 0.0 for w in omega]
    if adv_norm:
        sel = [a for a, m in zip(adv, mask) if m]
        if sel:
            mu = sum(sel) / len(sel)
            sd = (sum((x - mu) ** 2 for x in sel) / len(sel)) ** 0.5
            adv = [(x - mu) / (sd + 1e-6) for x in adv]
    if correct and correct_scale != 1.0:
        adv = [correct_scale * x for x in adv]
    if lambda_acc and acc_adv is not None:
        acc = lambda_acc * float(acc_adv)
        adv = [x + acc for x in adv]
    return adv, mask
