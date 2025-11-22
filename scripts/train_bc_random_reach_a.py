#!/usr/bin/env python3
# scripts/train_bc_random_reach_a.py
import argparse, numpy as np, torch
from controller.hybrid_bc_a import RLPolicyParams, RLPolicy, BehaviorCloner

def parse_hidden(s: str):
    return tuple(int(x) for x in s.split(","))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--hidden", type=str, default="256,256")
    ap.add_argument("--out", type=str, default="models/random_reach_bc_a.pt")
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    O, A = data["O"], data["A"]
    obs_dim, act_dim = O.shape[1], A.shape[1]
    hidden = parse_hidden(args.hidden)
    print(f"[BC-a] dataset loaded: O={O.shape} A={A.shape} (obs_dim={obs_dim}, act_dim={act_dim})")

    ppi = RLPolicyParams(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, device=args.device)
    pi  = RLPolicy(ppi)
    bc  = BehaviorCloner(pi, lr=args.lr, wd=args.wd)

    # holdout split
    N = O.shape[0]; i_val = int(0.9 * N)
    Otr, Atr = O[:i_val], A[:i_val]
    Oval, Aval = O[i_val:], A[i_val:]

    # whitening: prefer dataset's saved stats if available
    if "mean" in data.files and "std" in data.files:
        mean = data["mean"]; std = data["std"]
    else:
        mean = O.mean(axis=0); std = np.clip(O.std(axis=0), 1e-6, None)
    pi.mean = torch.tensor(mean, dtype=torch.float32, device=pi.device)
    pi.std  = torch.tensor(std , dtype=torch.float32, device=pi.device)

    # train
    bc.fit(Otr, Atr, epochs=args.epochs, batch_size=args.batch_size, shuffle=True)

    # quick val loss
    with torch.no_grad():
        import torch.nn as nn
        loss = nn.MSELoss()
        x = torch.tensor((Oval - pi.mean.cpu().numpy()) / (pi.std.cpu().numpy() + 1e-6),
                         dtype=torch.float32, device=pi.device)
        y = torch.tensor(Aval, dtype=torch.float32, device=pi.device)
        yhat = pi.model(x)
        val = float(loss(yhat, y).cpu().numpy())
    print(f"[BC-a] val MSE={val:.6f}")

    pi.save(args.out)
    print(f"[BC-a] saved: {args.out}")

if __name__ == "__main__":
    main()
