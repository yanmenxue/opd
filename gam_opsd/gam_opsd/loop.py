"""轮次编排：rollout -> grade -> train，各阶段独立进程，显存完全释放。

用法：
  python -m gam_opsd.loop --config configs/base.yaml --rounds 20
  python -m gam_opsd.loop --config configs/base.yaml --resume   # 跳过已完成轮次
                                                                # 与半完成轮次已落盘的阶段
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .common import load_config, round_dir


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    # 本节点 RoCE/libmlx5 与 NCCL 2.28 不兼容（ncclNetPluginInit 段错误）；
    # 单机训练不走 IB，禁用后回退 P2P/SHM。可被外部环境变量覆盖。
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # 大 [B,T,V] logits 张量反复分配导致预留碎片（曾达 6GB reserved-unallocated）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="round_dir/_done 存在则整轮跳过；rollouts/graded 已落盘则跳过对应阶段")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_gpu = args_n_gpu()
    n_rounds = args.rounds if args.rounds is not None else cfg["rounds"]["num_rounds"]

    for k in range(args.start, args.start + n_rounds):
        rd = round_dir(cfg["work_dir"], k)
        if args.resume and (rd / "_done").exists():
            print(f"\n===== ROUND {k} ===== [resume] complete, skip", flush=True)
            continue
        print(f"\n===== ROUND {k} =====", flush=True)
        if not (args.resume and (rd / "rollouts.jsonl").exists()):
            run([sys.executable, "-m", "gam_opsd.phase1_rollout", "--config", args.config,
                 "--round", str(k)])
        if not (args.resume and (rd / "graded.jsonl").exists()):
            run([sys.executable, "-m", "gam_opsd.phase2_grade", "--config", args.config,
                 "--round", str(k)])
        run([sys.executable, "-m", "torch.distributed.run",
             f"--nproc_per_node={n_gpu}", "-m", "gam_opsd.phase3_train",
             "--config", args.config, "--round", str(k)])
        (rd / "_done").touch()

        ev = cfg["rounds"].get("eval_every", 0)
        if ev and (k + 1) % ev == 0:
            run([sys.executable, "-m", "gam_opsd.eval_acc", "--config", args.config])


def args_n_gpu():
    import torch
    return torch.cuda.device_count()


if __name__ == "__main__":
    main()
