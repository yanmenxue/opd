"""CND 步级加权核心（纯 python，可单测）。

训练形态：spec 的直接加权 KL 损失需要在训练循环内做教师前向；本管线采用
离线打分 + 单 epoch REINFORCE 的等价代理：

    A_t = w_kl · ω_t · (−RKL_t) + λ·A_acc，其中 ω_t = w_k / |S_k|。

  N_k ∈ [0,1]（phase2 落盘：步骤分支 MC 的答案分布熵，N = 1 − H/log B）
  w = softmax(α·N_k)——关键决策点强蒸馏，松散步骤弱蒸馏
"""


def _softmax(xs):
    import math
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def cnd_step_weights(necessity, spans, alpha=1.0):
    """CND：softmax(α·N_k) → ω_t。N_k=None 的步骤不进 softmax、不受监督。

    返回 (omega, stats)。omega 长度 = spans[-1][1]（= 响应 token 数）。
    """
    n = spans[-1][1] if spans else 0
    if not spans:
        return [0.0] * n, {"kept": 0}
    kept = [k for k, v in enumerate(necessity) if v is not None]
    if not kept:
        return [0.0] * n, {"kept": 0}
    w = _softmax([alpha * necessity[k] for k in kept])
    omega = [0.0] * n
    for wi, k in enumerate(kept):
        s, e = spans[k]
        for t in range(s, e):
            omega[t] = w[wi] / (e - s)
    return omega, {"kept": len(kept),
                   "w": {k: round(x, 4) for k, x in zip(kept, w)}}


def omega_advantage(rkl, omega, w_kl=1.0, adv_norm=True,
                    correct=False, correct_scale=1.0, acc_adv=None, lambda_acc=0.0):
    """ω 加权优势：A_t = w_kl·ω_t·(−RKL_t)（+ λ·A_acc 全向量）。

    - correct_scale 是常因子，在归一化之后施加（z-norm 会精确消掉 norm 前
      的常数因子）；
    - adv_norm 在 ω>0 的 token 上标准化（保持与 baseline 臂可比的梯度尺度）。
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
