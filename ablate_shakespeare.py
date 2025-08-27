# ablate_shakespeare.py
import os, math, time, argparse, itertools
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig  # Fe‑RoPE + PDS attention + (optional) MFS + AE :contentReference[oaicite:2]{index=2}

# ---------- data ----------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def make_charset(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def split_train_val(enc, val_frac=0.1):
    n = len(enc); n_train = int(n * (1.0 - val_frac))
    return enc[:n_train], enc[n_train:]

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size]     for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------- eval (LM-only) ----------
@torch.no_grad()
def eval_lm_only(model, data, block_size, batch_size, steps, device):
    """
    LM-only NLL on held-out split. Robust to either:
      - MFS head (returns LOG-PROBS when configured that way), or
      - fallback linear head (returns logits).
    """
    model.eval()
    tot = 0.0; cnt = 0
    for _ in range(steps):
        x, y = get_batch(data, block_size, batch_size, device)
        logits, _ = model(x, targets=None)
        # Detect log-probs vs logits by exp-sum ~ 1.0
        s = torch.exp(logits[0,0]).sum().item()
        V = logits.size(-1)
        if abs(s - 1.0) < 1e-3:   # log-probs
            nll = F.nll_loss(logits.view(-1, V), y.view(-1), ignore_index=-1, reduction='mean')
        else:                      # logits
            nll = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1, reduction='mean')
        tot += float(nll); cnt += 1
    return tot / max(1, cnt)

def bpc(nll): return nll / math.log(2.0)
def ppl(nll): return math.exp(nll)

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/shakespeare_char/input.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=600, help="train steps per trial (keep small for grid)")
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--emb", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--ferope_m", type=int, default=32)
    p.add_argument("--seed", type=int, default=1337)
    # grid
    p.add_argument("--rings_grid", type=int, nargs="+", default=[1,2])
    p.add_argument("--mfs_grid", type=int, nargs="+", default=[1,0])
    p.add_argument("--ae_grid", type=int, nargs="+", default=[1,0])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    text = load_text(args.data)
    stoi, _ = make_charset(text)
    enc = encode(text, stoi)
    train_enc, val_enc = split_train_val(enc, val_frac=0.1)
    V = len(stoi)

    trials = list(itertools.product(args.rings_grid, args.mfs_grid, args.ae_grid))
    results = []

    for rings, use_mfs, use_ae in trials:
        print(f"\n=== Trial: rings={rings}  MFS={use_mfs}  AE={use_ae} ===")
        cfg = GPTConfig(
            block_size=args.block, vocab_size=V,
            n_layer=args.layers, n_head=args.heads, n_embd=args.emb,
            dropout=0.0, bias=True,
            # Fe‑RoPE + PDS anchors/mask inside attention 
            ferope_anchor_relative=True, ferope_m=args.ferope_m,
            use_block_sparse=True, neighbor_rings=rings,
            anchor_min_gap=max(32, args.block // 8), anchor_jitter=max(4, args.block // 64),
            # MFS head & AE head toggles 
            tie_weights=True, use_mfs_head=bool(use_mfs), mfs_K=3, mfs_P=4, mfs_lowrank_r=0,
            add_l4_losses=bool(use_ae), l4_loss_weight=0.1,
        )
        model = GPT(cfg).to(device)
        opt = model.configure_optimizers(weight_decay=args.wd, learning_rate=args.lr,
                                         betas=(0.9,0.95),
                                         device_type="cuda" if device.type=="cuda" else "cpu")

        # warmup→cosine
        def lr_lambda(step):
            if step < args.warmup: return max(1e-8, (step + 1)/max(1, args.warmup))
            t = (step - args.warmup)/max(1, args.steps - args.warmup)
            return max(args.min_lr/args.lr, 0.5*(1.0 + math.cos(math.pi*t)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        model.train()
        for step in range(1, args.steps+1):
            x, y = get_batch(train_enc, args.block, args.batch, device)
            opt.zero_grad(set_to_none=True)
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step()
            if step % 100 == 0:
                print(f"  step {step:4d}/{args.steps}  loss {float(loss):.4f}")

        val_lm = eval_lm_only(model, val_enc, args.block, args.batch, args.eval_batches, device)
        results.append(dict(rings=rings, mfs=use_mfs, ae=use_ae,
                            val_nll=val_lm, bpc=val_lm/math.log(2.0), ppl=math.exp(val_lm)))

    # print table (sorted by val_nll)
    print("\n=== Ablation results (sorted by val NLL) ===")
    results.sort(key=lambda r: r["val_nll"])
    header = f"{'rings':>5}  {'MFS':>3}  {'AE':>2}  {'val_nll':>8}  {'bpc':>6}  {'ppl':>6}"
    print(header); print("-"*len(header))
    for r in results:
        print(f"{r['rings']:5d}  {r['mfs']:>3d}  {r['ae']:>2d}  {r['val_nll']:8.4f}  {r['bpc']:6.3f}  {r['ppl']:6.2f}")

if __name__ == "__main__":
    main()
