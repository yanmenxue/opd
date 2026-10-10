# GAM-OPSD：Credit-Dispersion-aware Granularity-Adaptive On-Policy Self-Distillation（论文三）

OPSD 蒸馏信号在**粒度选择**上有结构性缺陷：token 级信用分配在"弥散信用"状态下存在统计效率的二次惩罚——推理操作的回报效应分散在步骤内 m 个 token 上时（每 token 效应量级 Θ(|A_step|/m)），区分有益/有害 token 所需 MC 采样预算与 m 的平方成正比（Θ(m²)），步骤级只需常数预算。现有方法粒度静态（全 token 级或全步骤级），没有工作按位置信用弥散度动态路由。GAM-OPSD 用**局部窗口内师生对数比的归一化方差**度量信用弥散度，弥散处聚合到步骤级、集中处保持 token 级。

## 流水线

```
Phase 0  probe    phase0_probe.py     难度探针（基座 pass@4 ∈ [1,3] 过滤中难度带）
Phase 1  rollout  phase1_rollout.py   学生每题采 G 条（vLLM）
Phase 2  grade    phase2_grade.py     逐 token 师生对数比 a_t = log π_T − log π_S
                                       + token 步骤边界 + 组内答案优势
Phase 3  train    phase3_train.py     FSDP；route 粒度路由；L_GAM 混合粒度 REINFORCE；
                                       轮级路由诊断 → gam_diag.jsonl
loop.py                轮次编排（各阶段独立进程，显存完全释放）
route.py               GAM 核心（纯 python 可单测）：弥散度/路由/混合优势
eval_acc.py            准确率（MATH-500 按 level 分层报告）
```

## 核心量

- **信用弥散度** `a_k = log π_T(y_k|x,y*,·) − log π_S(y_k|·)`，
  `Disp_t = Var_{k∈W_t}[a_k] / (Var[a]+ε)`（W_t = [t−w, t+w] 边界截断，两级均总体方差，前缀和 O(n)）。Disp 高 = 分歧均匀弥散在窗口内（弥散信用，token 级信号不可靠）；Disp 低 = 分歧集中在少数位置（集中信用，token 级可靠）；
- **粒度路由** `granularity_t = step if Disp_t > τ else token`（τ 默认 0.5）。损失以步骤为单位，按步骤聚合判定：**步骤 k 弥散 ⟺ 步内 mean Disp > τ**；
- **步骤级效应求和而非平均** `A_step = Σ_{t∈S_k} a_t`：求和让信号从 |A_step|/m 恢复到 |A_step|，噪声 std 只增长 √m 倍，**SNR 改善 √m**；
- **总损失** `L_GAM = Σ_{t: token 级} ℓ_t^token + λ_step Σ_{k: 弥散步} ℓ_k^step`。训练形态沿用本仓库离线打分 + REINFORCE 代理架构，per-token 形式：
  - token 级位置 `A_t = w_kl·a_t`（E_{y~π_S}[a_t] = −KL_t，单样本前向 KL 代理；符号自带方向：>0 = 教师更确信采样 token → 提升）；
  - 弥散步内平摊 `A_t = λ_step·w_kl·A_step/|S_k|`（等价于 ℓ_k^step = −A_step·mean_{t∈S_k} log π）；
  - `adv_norm` 仅在 token 分支内 z-norm（a 的符号携带方向，全量 z-norm 会抹掉均值中的蒸馏压力）。

**一个旋钮覆盖全部主对照臂**（`route.mode`）：`token` = 标准 OPSD / `step` = 全步骤级 OPSD / `dynamic` = GAM；纯 GRPO = `train.w_kl: 0.0` + `arm: opd_acc`。固定粒度臂与 GAM 用同一 a_t 估计器，粒度对比不被信号族差异混淆。

## 与论文一/二的边界

本包**不含** IER/双轴精准干预（`../sropsd/`）与因果干预/分支 MC 步骤信用（`../cnd/`）；本方法的步骤级信号来自**同一 a_t 信号的聚合层级切换**，而非新的步级评分器。三包共享的只有算法无关工具（common/prompts/data/rollout/eval_acc）。

## 快速开始

```bash
pip install -r requirements.txt
# 网络受限时先设 HF 镜像：export HF_ENDPOINT=https://hf-mirror.com

# 0) 构建训练题：难度探针（基座 pass@4 过滤中难度带）→ runs/main/prompts.jsonl
python -m gam_opsd.phase0_probe --config configs/base.yaml

# 1-3) 轮次循环
python -m gam_opsd.loop --config configs/base.yaml --rounds 20

# 评测（data_eval: math500 主评测（按 level 分层）| aime24/aime25 OOD | gsm8k_test 辅助）
python -m gam_opsd.eval_acc --config configs/base.yaml --model <student_path>
```

冒烟（0.5B 自蒸馏 2 轮）：`python -m gam_opsd.loop --config configs/smoke.yaml --rounds 2`
单测：`python -m gam_opsd.test_route`

## 实验设计

- **主对照**：标准 OPSD（`route.mode: token`）/ 全步骤级 OPSD（`route.mode: step`）/ GAM-OPSD（`route.mode: dynamic`）/ 纯 GRPO（再加 `train.w_kl: 0.0`）；
- **消融**：τ 敏感性（0.3/0.5/0.7）、窗口 w（8/16/32）、强制固定粒度（token/step）vs 动态路由；
- **关键诊断**（`runs/*/gam_diag.jsonl`，逐轮追加）：弥散度分布（mean/median/p25/p75）随训练的漂移、步级路由占比、corr(Disp_t, 步长)——预期弥散度与步骤内 token 数正相关；
- **预期发现**：标准 OPSD 在弥散信用区域浪费梯度；GAM-OPSD 在这些区域切换步骤级监督后更优；训练后期弥散度分布左移（师生分歧收敛到少数位置）。

## 已知简化（spec → 实现）

1. REINFORCE 代理而非直接 k3 蒸馏（教师不进训练循环，离线打分架构，同论文一/二）；
2. Disp 的分母 Var[a] 取**全响应方差**（spec 未指明归一化域；窗口方差同域可得 Disp≈1 的"均匀弥散"标定点，且 a 全相等时 Disp≡0 安全退化）；
3. spec 的 per-token 粒度标签落到步级损失时，按**步内 mean Disp > τ** 聚合判定（步骤级分支以整步为单位：步内全部 token 的 a 求和、优势整步平摊）；
4. 所有臂（含标准 OPSD 臂）统一用 a_t 单样本 log-ratio 估计器（期望等于 −RKL 的 REINFORCE 代理），保证粒度对比单变量；E[a] = −KL 在无穷样本下与 sropsd 的 −RKL 臂同期望、方差不同——方差差异正是本方法的研究对象；
5. 弥散度窗口与步骤切分均基于 token 序列；步骤切分 regex + tokenizer 对齐，失败回退固定窗口；
6. checkpoint 单文件 safetensors（bf16），存 `train.ckpt_dir`（与初始模型路径分离，教师始终加载原始 `teacher_model`）；train_log / gam_diag 逐轮追加（区别于兄弟包的覆盖写）。
