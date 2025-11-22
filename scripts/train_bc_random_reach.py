#!/usr/bin/env python3
import argparse, os, numpy as np, torch
from controller.hybrid_mpc_rl import RLPolicy, RLPolicyParams, BehaviorCloner

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=str, default="256,256")
    p.add_argument("--save", type=str, default="models/random_reach_bc.pt")
    return p.parse_args()

def main():
    args = parse()
    data = np.load(args.dataset)
    O, A = data["O"].astype(np.float32), data["A"].astype(np.float32)   # O: (N, 14)
    N = O.shape[0]
    n_val = max(10000, int(0.1 * N))
    idx = np.random.permutation(N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Otr, Atr = O[tr_idx], A[tr_idx]
    Oval, Aval = O[val_idx], A[val_idx]

    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    ppi = RLPolicyParams(obs_dim=O.shape[1], hidden=hidden, device=args.device)
    pi = RLPolicy(ppi)
    bc = BehaviorCloner(pi, lr=args.lr, wd=args.wd)

    # simple cosine lr schedule
    opt = bc.opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # training loop with printed metrics
    for ep in range(1, args.epochs + 1):
        bc.fit(Otr, Atr, epochs=1, batch_size=args.batch_size, shuffle=True)
        with torch.no_grad():
            # eval
            dev = pi.device
            Om = torch.tensor(Otr, dtype=torch.float32, device=dev)
            Am = torch.tensor(Atr, dtype=torch.float32, device=dev)
            mean = Om.mean(dim=0); std = Om.std(dim=0).clamp_min(1e-6)
            yhat_tr = pi.model((Om - mean)/(std+1e-6))
            tr_loss = torch.mean((yhat_tr - Am)**2).item()

            Ov = torch.tensor(Oval, dtype=torch.float32, device=dev)
            Av = torch.tensor(Aval, dtype=torch.float32, device=dev)
            yhat_val = pi.model((Ov - mean)/(std+1e-6))
            val_loss = torch.mean((yhat_val - Av)**2).item()

        sched.step()
        print(f"[{ep:03d}/{args.epochs}] train={tr_loss:.6f}  val={val_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    pi.save(args.save)
    print(f"[BC] saved: {args.save}")

if __name__ == "__main__":
    main()
