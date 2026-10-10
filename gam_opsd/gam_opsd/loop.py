"""轮次编排：rollout -> grade -> train，各阶段独立进程，显存完全释放。

用法：
  python -m gam_opsd.loop --config configs/base.yaml --rounds 20
"""
import argparse
import subprocess
import sys

from .common import load_config


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_gpu = args_n_gpu()
    n_rounds = args.rounds if args.rounds is not None else cfg["rounds"]["num_rounds"]

    for k in range(args.start, args.start + n_rounds):
        print(f"\n===== ROUND {k} =====", flush=True)
        run([sys.executable, "-m", "gam_opsd.phase1_rollout", "--config", args.config,
             "--round", str(k)])
        run([sys.executable, "-m", "gam_opsd.phase2_grade", "--config", args.config,
             "--round", str(k)])
        run([sys.executable, "-m", "torch.distributed.run",
             f"--nproc_per_node={n_gpu}", "-m", "gam_opsd.phase3_train",
             "--config", args.config, "--round", str(k)])

        ev = cfg["rounds"].get("eval_every", 0)
        if ev and (k + 1) % ev == 0:
            run([sys.executable, "-m", "gam_opsd.eval_acc", "--config", args.config])


def args_n_gpu():
    import torch
    return torch.cuda.device_count()


if __name__ == "__main__":
    main()
