# Causal-Credit OPSD：基于因果干预的步骤级信用分配蒸馏（论文二）

OPSD 的被忽视缺陷是**步骤级信用分配缺失**：token 级监督隐含"每步同等重要"的假设，但分歧大 ≠ 因果重要——一个步骤可以与教师分歧很大但从它出发多路可达答案（因果解耦），也可以分歧不大但移除后答案必然改变（因果关键）。本工作用**反事实干预信号替代"分歧幅度"**作步骤级信用分配的调制器。算法详见 `../DESIGN_two_works.md` §2.7。

## 流水线

```
Phase 1  rollout     phase1_rollout.py     学生每题采 G 条（vLLM）
Phase 2  intervene   phase2_intervene.py   正确 rollout（长度>300tok）每步：
                                           ① 学生改写出语义矛盾、同粒度的 s_k'（V 变体）
                                           ② 从干预点重生成后续推理 → 答案 a*
                                           ③ φ_k = mean_V[1 − Sim(a, a*)] → cc.jsonl
Phase 3  grade       phase3_grade.py       逐 token RKL + 步骤边界 + 组内答案优势
Phase 4  train       phase4_train.py       w=softmax(α·φ_k) → A_t = w_kl·ω_t·(−RKL_t) + λ·A_acc
loop.py                   轮次编排（各阶段独立进程，显存完全释放）
eval_acc.py               准确率
```

## 核心机制

- **因果敏感度 φ_k**：干预类型三选一（消融轴）——`negate` 逻辑反转 / `factual` 事实反转（翻数值）/ `premise` 前提否定；Sim 用 math_verify 等价（精确匹配）。φ 高 = 因果必要（关键决策步骤），φ 低 = 因果解耦（冗长解释、重复验证、装饰性推导）；
- **变体数 V**：spec 的单次干预是方差上界（φ 退化为 0/1 二值）；默认 V=2 作方差缩减，可消融回 1。变体全部无答案 → φ=None（不进 softmax、不受监督）；
- **与 CND 分支 MC 的区别**（现为消融臂 `cnd/phase2_branch.py`）：分支 MC 测"该状态多路可达性"（中性续写的答案分布熵 N_k），本方法测"答案对该步内容的因果敏感度"（对抗替换的翻转率 φ_k）——前者"测路的宽度"，后者"测路的方向依赖"；
- **与 GraphOPD 的区别**：GraphOPD 的依赖图来自结构化环境记录；本方法的信号来自反事实干预的答案变化，无需环境；
- 训练形态：REINFORCE 代理 A_t = w_kl·ω_t·(−RKL_t) + λ·A_acc（直接加权 KL 损失需教师进训练循环，留作消融）。

## 与论文一的边界

本包不含 token 级选择/IER（那是 `../sropsd/` 的精准干预轴）。两包共享的只有算法无关工具。信号源消融（φ vs N_k vs 均匀）在本包内用 `select.mode` 切换。

## 快速开始

```bash
pip install -r requirements.txt
# 网络受限时先设 HF 镜像：export HF_ENDPOINT=https://hf-mirror.com

# 0) 构建训练题：难度探针（基座 pass@4 过滤中难度带）→ runs/main/prompts.jsonl
python -m cnd.phase0_probe --config configs/base.yaml

# 1-4) 轮次循环（含 phase2_intervene）
python -m cnd.loop --config configs/base.yaml --rounds 20

# 评测（data_eval: math500 主评测 | aime24/aime25/amc23 OOD | gsm8k_test 辅助）
python -m cnd.eval_acc --config configs/base.yaml --model <student_path>
```

冒烟（0.5B 自蒸馏 2 轮）：`python -m cnd.loop --config configs/smoke.yaml --rounds 2`
单测：`python -m cnd.test_weights`

## 实验设计

- **主对照**：标准 OPSD（`mode: none`）/ CND 分支 MC（`mode: cnd` + `cnd.enable: true`）/ Causal-Credit（`mode: causal_credit`）/ 纯 GRPO（再加 `w_kl: 0.0`）；
- **消融**：干预类型（negate/factual/premise）、α 锐度、V=1 vs 2、与 token 级分歧信号（sropsd 包的 RKL/IER）的对比；
- **关键诊断**：对答案正确的 rollout 施加干预后答案改变的比率——Causal-Credit 训练中应保持稳定或上升（不损害因果敏感度）；
- **预期发现**：标准 OPSD 在 AIME 上可能因蒸馏因果解耦步骤而低于纯 GRPO；Causal-Credit 显著优于标准 OPSD。

## 已知简化

1. Sim 为二值等价（math_verify），非数值近似距离——φ 在 V=1 时退化为 0/1；
2. REINFORCE 代理而非直接 k3 蒸馏（离线打分架构）；
3. 干预改写由学生自任（spec），改写质量无外部校验——病态改写（空/超长）被截断或跳过；
4. 步骤切分 regex + tokenizer 对齐，失败回退固定窗口；checkpoint 存 `train.ckpt_dir`（与初始模型分离）。
