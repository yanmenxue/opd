"""精准干预的 token 级选择核心（纯 python，可单测）。

轴一 usefulness：逐 token RKL（教师位移幅度）。
轴二 reliability：rel = (1/(1+H))^γ。A_t 是确定性前向量、无采样噪声；
真实噪声源是 REINFORCE 单样本 score-function 方差 + 前缀随机性传导，
categorical 分布下熵是其 Fisher 方差的标量代理——高熵位置即使分歧大，
单样本梯度估计也不稳，慎蒸。（SR-OPSD 主方法用 phase2 的 IER 替代本轴；
本模块保留熵代理作为廉价消融。）
步轴：mean RKL < τ_align 且 mean H < τ_unc 的步骤"退役"（学生已确定且与
教师对齐，继续监督只会锁死教师的表述偏好）。
方向感知：学生答案正确的 rollout，师生分歧更可能是"教师表述偏好 vs
学生捷径"而非知识差 → KL 缩放 correct_scale（在 train 优势归一化之后
施加；仅 opd_acc 等带结果信号的臂有意义）。
"""


def reliability(ent: float, gamma: float = 1.0) -> float:
    """轴二：学生熵 → 梯度可靠性权重 ∈ (0,1]。"""
    return (1.0 / (1.0 + max(ent, 0.0))) ** gamma


def retire_step(mean_rkl: float, mean_ent: float,
                tau_align: float, tau_unc: float) -> bool:
    """步退役判据：与教师已对齐 且 学生已确定。"""
    return mean_rkl < tau_align and mean_ent < tau_unc


def select_mask(rkl, ent, step_spans=None, mode="none", budget=1.0, gamma=1.0,
                tau_align=None, tau_unc=None, min_tokens=1):
    """返回 (mask, stats)。mask 与 rkl 等长（bool list）。

    mode:
      none        全量监督（baseline）
      usefulness  token top-b（按 RKL）
      dual        token top-b（按 RKL·(1/(1+H))^γ）
      step_retire 步退役，保留步内全量（budget<1 时步内再做 usefulness top-b）
      dual_step   步退役 + 步内 dual top-b
    budget 相对响应全长（跨模式可比）；入选集允许混入零分 token
    （adv=0 无梯度，只影响损失分母），保证 min_tokens 下限成立。
    stats: frac 入选比例 / retired 退役步数 / kl_coverage 入选 token 承载的 KL 占比。
    """
    n = len(rkl)
    if n == 0 or mode in ("none", None):
        return [True] * n, {"frac": 1.0, "retired": 0, "kl_coverage": 1.0}

    total_kl = sum(rkl) or 1e-12
    spans = step_spans or [(0, n)]

    # 步级退役
    kept, retired = [], 0
    step_gate = mode in ("step_retire", "dual_step") and tau_align is not None
    for s, e in spans:
        if step_gate:
            seg = rkl[s:e]
            mr = sum(seg) / len(seg)
            me = sum(ent[s:e]) / len(seg)
            if retire_step(mr, me, tau_align, tau_unc or 0.0):
                retired += 1
                continue
        kept.append((s, e))
    kept_n = sum(e - s for s, e in kept)
    if kept_n == 0:
        return [False] * n, {"frac": 0.0, "retired": retired, "kl_coverage": 0.0}

    # token 级 top-b（kept 区内）
    quota = min(kept_n, max(min_tokens, int(round(budget * n))))
    score_mode = "dual" if mode in ("dual", "dual_step") else "usefulness"
    cand = []
    for s, e in kept:
        seg = range(s, e)
        scores = [k * reliability(h, gamma) for k, h in zip(rkl[s:e], ent[s:e])] \
            if score_mode == "dual" else list(rkl[s:e])
        cand.extend(zip(seg, scores))
    cand.sort(key=lambda x: -x[1])
    chosen = {i for i, _ in cand[:quota]}

    mask = [i in chosen for i in range(n)]
    stats = {
        "frac": len(chosen) / n,
        "retired": retired,
        "kl_coverage": sum(rkl[i] for i in chosen) / total_kl,
    }
    return mask, stats
