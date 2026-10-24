#!/usr/bin/env bash
# =============================================================================
# GAM-OPSD 重跑流水线
#   预算 4096 / thinking 保持开 / eval greedy / 丢弃未框答案(drop_unboxed)
#
# 用法：
#   bash run_pipeline.sh smoke          # 自包含冒烟(~10 分钟)：先验代码改动
#   bash run_pipeline.sh probe          # phase0 造题池(~7 小时，只跑一次)
#   bash run_pipeline.sh stage1         # ★ 一条命令：probe -> base 基线 -> main/grpo/opd
#   bash run_pipeline.sh arms main      # 跑指定臂，可跟多个（base 是 main 的别名）
#   bash run_pipeline.sh all            # 5 臂全跑(主实验)
#   bash run_pipeline.sh status         # 各臂进度
#   bash run_pipeline.sh report         # 汇总 eval_summary.jsonl
#
# 环境变量：
#   PY=<python>   覆盖解释器路径
#   ROUNDS=<n>    覆盖轮数(默认取 configs 里的 num_rounds=20)
#   RESUME=1      给 loop 加 --resume：跳过已落盘的 rollout/graded 与已完成的整轮
#   DETACH=1      arms/all/stage1 整串丢后台（顺序执行，日志 runs/_pipeline.log）
#                 注意 smoke / probe 不受影响，仍是前台
#
# 注意：
#   * configs/base.yaml 是「模板」，实际运行的是生成出来的 configs/<arm>.yaml
#     （每个臂有独立的 work_dir 与 ckpt_dir，互不覆盖）
#   * 所有臂共用同一份 runs/main/prompts.jsonl 题池，禁止各臂重跑 phase0
#     （题池是按基座 pass@4 筛的中难度带，重跑会得到不同的题目集合）
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"          # 必须：work_dir / ckpt_dir 都是相对 cwd 的路径
PY="${PY:-/root/anaconda3/envs/python12/bin/python}"
ROUNDS="${ROUNDS:-}"
export NCCL_IB_DISABLE=1                       # 本机 libmlx5 过旧，NCCL 走 IB 会段错误
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARMS=(main grpo token step opd)

# ---------------------------------------------------------------- 生成配置
gen_configs() {
  $PY - <<'PYEOF'
import pathlib
base = pathlib.Path('configs/base.yaml').read_text(encoding='utf-8')

SPECS = {
    # 主臂：动态路由 + 答案优势
    'main':  [],
    # 对照臂：关掉蒸馏分支（纯 RLVR/GRPO）—— 回答「涨幅是否 RL 就能拿到」
    'grpo':  [('  w_kl: 1.0',
               '  w_kl: 0.0               # 对照臂：关掉蒸馏分支（纯 RLVR/GRPO）')],
    # 对照臂：全 token 级（标准 OPSD）
    'token': [('  mode: "dynamic"',
               '  mode: "token"        # 对照臂：全 token 级')],
    # 对照臂：全步骤级
    'step':  [('  mode: "dynamic"',
               '  mode: "step"         # 对照臂：全步骤级')],
    # 免答案臂（论文卖点之一）
    'opd':   [('  arm: "opd_acc"', '  arm: "opd"            # 免答案臂'),
              ('  lambda_acc: 1.0', '  lambda_acc: 0.0      # 免答案：关掉答案优势')],
    # 冒烟：极小规模、自包含（自己造题池）
    'smoke': [('  max_questions: 4000', '  max_questions: 16'),
              ('  prompts_per_round: 256', '  prompts_per_round: 16'),
              ('  G: 8', '  G: 2'),
              ('  max_questions: 500', '  max_questions: 16'),
              ('  eval_every: 5', '  eval_every: 1')],
}

for arm, edits in SPECS.items():
    s = base.replace('work_dir: "./runs/main"', 'work_dir: "./runs/%s"' % arm, 1)
    s = s.replace('ckpt_dir: "./ckpt"', 'ckpt_dir: "./runs/%s/ckpt"' % arm, 1)
    for old, new in edits:
        assert old in s, '[%s] 未匹配: %r' % (arm, old)
        s = s.replace(old, new, 1)
    p = pathlib.Path('configs/%s.yaml' % arm)
    if p.exists() and p.read_text(encoding='utf-8') == s:
        continue
    p.write_text(s, encoding='utf-8')
    print('  生成 configs/%s.yaml   work_dir=./runs/%s  ckpt=./runs/%s/ckpt' % (arm, arm, arm))
PYEOF
}

# ---------------------------------------------------------------- 跑一个臂
ensure_prompts() {          # 题池只造一次，其余臂复用
  local arm=$1
  mkdir -p "runs/$arm"
  [[ -f "runs/$arm/prompts.jsonl" ]] && return 0
  [[ -f runs/main/prompts.jsonl ]] || {
    echo "缺少 runs/main/prompts.jsonl —— 先跑: bash $0 probe" >&2; exit 1; }
  cp runs/main/prompts.jsonl "runs/$arm/prompts.jsonl"
  echo "[$arm] 复用 runs/main/prompts.jsonl（各臂必须同题池）"
}

baseline() {                # 未训练的基座，作为所有臂的参照
  local m
  if [[ -s runs/base/eval_base.jsonl ]]; then
    echo "[base] runs/base/eval_base.jsonl 已存在，跳过（要重测就删掉它）"
    return 0
  fi
  m=$($PY -c "import yaml;print(yaml.safe_load(open('configs/main.yaml',encoding='utf-8'))['student_model'])")
  mkdir -p runs/base
  echo ""
  echo "=== [base] 基座评测（未训练）: $m ==="
  $PY -m gam_opsd.eval_acc --config configs/main.yaml --model "$m" \
    --out runs/base/eval_base.jsonl
}

_run_arm_fg() {             # 前台顺序跑一个臂
  local arm=$1
  if [[ "$arm" == "base" ]]; then arm=main; fi   # base.yaml 那一份 = 主臂
  local cfg="configs/$arm.yaml"
  [[ -f "$cfg" ]] || { echo "没有 $cfg，先跑 gen"; exit 1; }
  ensure_prompts "$arm"
  local extra=()
  if [[ -n "$ROUNDS" ]]; then extra+=(--rounds "$ROUNDS"); fi
  if [[ "${RESUME:-0}" == "1" ]]; then extra+=(--resume); fi   # 跳过已落盘的 rollout/graded
  echo ""
  echo "=== [$arm] $PY -m gam_opsd.loop --config $cfg ${extra[*]:-} ==="
  $PY -m gam_opsd.loop --config "$cfg" "${extra[@]}"
}

run_arms() {                # 顺序跑多个臂；DETACH=1 把整串丢后台（不会并发抢卡）
  local arms=("$@")
  if [[ "${DETACH:-0}" == "1" ]]; then
    mkdir -p runs
    ( for a in "${arms[@]}"; do _run_arm_fg "$a"; done ) > runs/_pipeline.log 2>&1 &
    echo "后台 pid=$!   日志: tail -f runs/_pipeline.log"
  else
    for a in "${arms[@]}"; do _run_arm_fg "$a"; done
  fi
}

# ---------------------------------------------------------------- 子命令
case "${1:-help}" in
  smoke)
    echo "== 冒烟：自包含（自己造 16 题池 + 1 轮训练），约 10 分钟 =="
    gen_configs
    $PY -m gam_opsd.phase0_probe --config configs/smoke.yaml
    ROUNDS="${ROUNDS:-1}"
    _run_arm_fg smoke
    echo ""
    echo "== 验收四件事 =="
    echo "  1) grade 打印 [grade] teacher -> cuda:0, student -> cuda:1（单卡直载）"
    echo "  2) grade 打印 '前向 N 条（跳过未框 M 条）' 且 M > 0"
    echo "  3) train 打印 '其中未框答案=' 且 > 0；有 frac_steps=/n_steps=/p25=/p75="
    echo "  4) runs/smoke/ckpt/round_000/ 存在（eval_every=1）"
    echo ""
    echo "顺手看一眼 grade 那一段的 wall time（改造前后对比），rollout 不会变快。"
    ;;

  probe)
    gen_configs
    echo "== phase0 造题池（max_questions=4000, G=4, 预算 4096，约 7 小时）=="
    $PY -m gam_opsd.phase0_probe --config configs/main.yaml
    echo "== 完成 -> runs/main/prompts.jsonl =="
    ;;

  arms)
    shift
    [[ $# -gt 0 ]] || { echo "用法: bash $0 arms main [grpo token step opd]"; exit 1; }
    gen_configs
    run_arms "$@"
    ;;

  all)
    gen_configs
    [[ -f runs/main/prompts.jsonl ]] || \
      { echo "先跑: bash $0 probe（题池只造一次）"; exit 1; }
    run_arms "${ARMS[@]}"
    ;;

  stage1)
    # 一条命令走完：造题池(若缺) -> 基座基线 -> main(base) / grpo / opd 三臂
    echo "阶段1：probe -> base 基线 -> main / grpo / opd"
    gen_configs
    if [[ -f runs/main/prompts.jsonl ]]; then
      echo "[题池] runs/main/prompts.jsonl 已存在，跳过 probe"
    else
      echo "== phase0 造题池（约 7 小时）=="
      $PY -m gam_opsd.phase0_probe --config configs/main.yaml
    fi
    baseline
    run_arms main grpo opd
    echo ""
    echo "== 阶段1 全部完成 =="
    bash "$0" report
    ;;

  status)
    $PY - <<'PYEOF'
import json, pathlib
for d in sorted(pathlib.Path('runs').glob('*')):
    if not d.is_dir():
        continue
    done = sum(1 for r in sorted(d.glob('round_*')) if (r / '_done').exists())
    snaps = len(list((d / 'ckpt').glob('round_*'))) if (d / 'ckpt').exists() else 0
    info = ''
    log = d / 'train_log.jsonl'
    if log.exists():
        rows = [json.loads(l) for l in open(log, encoding='utf-8')]
        if rows:
            r = rows[-1]
            info = ('round=%-3s loss=%+.4f n_samples=%-5s 未框丢弃=%-5s '
                    '路由比例=%-6s' % (r['round'], r['loss'], r['n_samples'],
                                    r.get('dropped_unboxed'), r.get('frac_step_tokens')))
    print('%-8s 完成 %2d 轮  ckpt快照 %d  %s' % (d.name, done, snaps, info))
PYEOF
    ;;

  report)
    $PY - <<'PYEOF'
import json, pathlib
found = False
for s in sorted(pathlib.Path('runs').glob('*/eval_summary.jsonl')):
    found = True
    for line in open(s, encoding='utf-8'):
        r = json.loads(line)
        print('%-8s %-28s acc=%.4f  未框率=%.3f  %s' % (
            s.parent.name, pathlib.Path(r['file']).name, r['acc'], r['unboxed_rate'],
            ' '.join('L%s:%.3f' % (k, v) for k, v in sorted(r['by_level'].items()))))
if not found:
    print('还没有 eval_summary.jsonl（评测每 5 轮一次，round 4 结束才有）')
PYEOF
    ;;

  *)
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
