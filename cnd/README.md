# CND：Causal-Necessity Distillation（论文二）

现有 PRM 回答"这一步对不对"，但无法区分**关键推理步骤**与**装饰性步骤**。CND 用步骤分支 MC 的答案分布熵识别真正决定答案的步骤（状态级因果结构，可被表面模式欺骗的反面），把蒸馏信号集中在这些步骤上。算法推导与实验矩阵见 `../DESIGN_two_works.md` §2.6。

## 流水线

```
Phase 1  rollout   phase1_rollout.py   学生每题采 G 条（vLLM）
Phase 2  branch    phase2_branch.py    正确 rollout 每步结尾续写 B 条 → P_k
                                       → N_k = 1 − H_smooth(P_k)/log B
Phase 3  grade     phase3_grade.py     逐 token RKL + 步骤边界 + 组内答案优势
Phase 4  train     phase4_train.py     FSDP；A_t = w_kl·ω_t·(−RKL_t) + λ·A_acc
loop.py                 轮次编排（各阶段独立进程，显存完全释放）
eval_acc.py             准确率
```

## 核心量

- **价值分布 P_k**：从步骤 s_k 末尾续写 B 条短 rollout 的答案经验分布；
- **因果必要性 N_k = 1 − H(P_k)/log B ∈ [0,1]**：N 高 = 从该步出发答案几乎唯一决定（关键决策点）；N 低 = 多路可达（松散步骤）。H_max = log B 而非 log|A|（后者随观测支持集变化，N 不可比）；
- **与 PRM 的本质区别**：PRM 学"步骤本身对不对"（状态级正确性），CND 用"从该步出发多少路走得通"（状态级因果结构，搜索出来的反事实信号）；
- 步权重 `w = softmax(α·N_k)`，`ω_t = w_k/|S_k|`；N=None 的步（续写全无答案）不进 softmax、不受监督；
- 训练 `A_t = w_kl·ω_t·(−RKL_t) + λ·A_acc`（spec 的加权 KL + λ·L_RL；REINFORCE 代理）。

## 与论文一的边界

本包**不含** token 级选择/IER/熵代理（那是 `../sropsd/` 的精准干预轴）；也**不含**原 mcpo 的教师轨迹闸门（phase2 verify / trust 已随 Design B 废弃）。两包共享的只有算法无关工具。

## 快速开始

```bash
pip install -r requirements.txt

# 0) 构建训练题
python -c "from cnd.data import build_prompts; from cnd.common import load_config; build_prompts(load_config('configs/base.yaml'))"

# 1-4) 轮次循环
python -m cnd.loop --config configs/base.yaml --rounds 20

# 评测
python -m cnd.eval_acc --config configs/base.yaml --model <student_path>
```

冒烟（0.5B 自蒸馏 2 轮）：`python -m cnd.loop --config configs/smoke.yaml --rounds 2`

## 实验设计（spec）

- **基线**：纯 GRPO（`arm: opd_acc` + `select.mode: none` + `w_kl: 0.0`）/ GRPO+PRM / MCTS 价值直接蒸馏（PRM 变体）/ 全量 OPSD（`mode: none` + `w_kl: 1.0`）；
- **预期发现**：显著降低"错误推理+碰巧正确"类样本的强化，不损害正确推理类准确率；
- B=4 vs B=8 作 ablation；忠实性三类样本评估集（对/碰巧对/算错）待实现（metrics 扩展）；
- 单测：`python -m cnd.test_weights`。

## 已知简化

1. "MCTS"操作化为步骤分支 MC（每步 B 条续写）；UCT 式干预搜索（删除/替换步）属扩展；
2. REINFORCE 代理而非直接 k3 蒸馏（离线打分架构）；
3. 步骤切分 regex + tokenizer 对齐，失败回退固定窗口；
4. checkpoint 单文件 safetensors（bf16），存 `train.ckpt_dir`（与初始模型路径分离，教师始终加载原始 `teacher_model`，不会被学生 checkpoint 遮蔽）。
