# SR-OPSD：Step-wise Reliability-Aware On-Policy (Self-)Distillation（论文一）

从"全面监督"到"精准干预"：师生分歧高度集中在极少数位置，均匀逐 token KL 监督冗余甚至有害。SR-OPSD 把监督选择上移到**步骤级**，并引入**梯度估计可靠性**作为第二筛选轴。算法推导与实验矩阵见 `../DESIGN_two_works.md` §1.7。

## 流水线

```
Phase 1  rollout   phase1_rollout.py   学生每题采 G 条（vLLM）
Phase 2  grade     phase2_grade.py     逐 token 四信号 + token 步骤边界
                                       rkl / entropy / IER / log-ratio / step_spans
Phase 3  train     phase3_train.py     FSDP；select 六模式；单 epoch REINFORCE
loop.py                 轮次编排（各阶段独立进程，显存完全释放）
eval_acc.py             准确率
```

## 核心量

- **轴一（有用性）** `U_k = mean_{t∈S_k}(log π_S − log π_T)`：学生过自信度（>0 = 学生比教师更确信自己的采样 token）；
- **轴二（可靠性）** `R_k = mean IER`，`IER = ‖g‖²/E‖ĝ‖²`：g = p_S⊙(ℓ−KL) 是 RKL 对学生 logits 的精确梯度，ĝ(y)=ℓ_y(e_y−p_S) 是其无偏单样本 score 估计，top-K 并集近似。**注意**：confident-extreme 处 g≡0——双轴实际选出"有质量散布 + 师生分歧"的位置，不是"自信但错"；
- 步权重 `w = softmax(α·U_k·R_k)`（可选门控 `[U_k>τ_U]·[R_k>τ_R]`），`ω_t = w_k/|S_k|`（Σ_t ω_t = 1/rollout）；
- 训练 `A_t = w_kl·ω_t·(−RKL_t)`（REINFORCE 代理；直接 k3 蒸馏需教师进训练循环，留作消融）。

token 级消融模式（`select.py`）：`usefulness`（top-b RKL）/ `dual`（RKL×(1/(1+H))^γ 熵代理）/ `step_retire` / `dual_step`。

## 与论文二的边界

本包**不含**任何一致性/忠实性闸门与步骤价值分布（原 mcpo 的 verify/trust/CND 已剥离）；步骤分支 MC（CND）在 `../cnd/`。两包共享的只有算法无关工具（common/prompts/data/rollout/eval_acc）。

## 快速开始

```bash
pip install -r requirements.txt

# 0) 构建训练题
python -c "from sropsd.data import build_prompts; from sropsd.common import load_config; build_prompts(load_config('configs/base.yaml'))"

# 1-3) 轮次循环
python -m sropsd.loop --config configs/base.yaml --rounds 20

# 评测
python -m sropsd.eval_acc --config configs/base.yaml --model <student_path>
```

冒烟（0.5B 自蒸馏 2 轮）：`python -m sropsd.loop --config configs/smoke.yaml --rounds 2`

## 配置要点（configs/base.yaml）

- `grade.teacher_sees_gold: true`：自蒸馏规范设定 π_T(·|x,y*,·)（师生同 backbone）；跨模型教师需同 tokenizer，否则改 false；
- **RLVR baseline**：`train.w_kl: 0.0` + `arm: opd_acc`；
- **预算扫描**：token 模式 `select.budget ∈ {0.001, 0.01, 0.05, 1.0}`；
- **必做 baseline**：`mode: none`（全量 OPSD）与 80/20 式高熵 masking（`usefulness` 换熵排序即可近似）；TIP 式"仅有用性"基线 = `mode: usefulness`；
- 单测：`python -m sropsd.test_select && python -m sropsd.test_weights`。

## 已知简化

1. REINFORCE 代理而非直接 k3 蒸馏（教师不进训练循环，离线打分架构）；
2. 步骤切分 regex + tokenizer 对齐，失败回退固定窗口；
3. checkpoint 单文件 safetensors（bf16），存 `train.ckpt_dir`（与初始模型路径分离，教师始终加载原始 `teacher_model`，不会被学生 checkpoint 遮蔽）；
4. IER 的 Σp² 在 top-K 并集上近似（低估尾部质量）。
