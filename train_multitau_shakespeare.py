import os
import math
import torch

from cortex.io_patches import TextSensor
from cortex.hexgrid import make_grid, build_adjacency
from torch.optim import Adam


# --------------------
# Data Loading
# --------------------
data_path = "data/shakespeare_char/input.txt"
if not os.path.exists(data_path):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    sample_text = (
        "From fairest creatures we desire increase,\n"
        "That thereby beauty's rose might never die.\n"
    )
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()
print(f"Data has {len(text)} characters.")

# Vocabulary
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} characters")

# Encode and split
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]
print(f"Train set length: {len(train_data)} tokens, Val set length: {len(val_data)} tokens")
block_size = min(128, len(train_data) - 1, len(val_data) - 1)



def get_batch(split: str, batch_size: int, block_size: int):
    ds = train_data if split == "train" else val_data
    max_start = len(ds) - (block_size + 1)
    if max_start <= 0:
        x = ds[:block_size].unsqueeze(0).repeat(batch_size, 1)
        y = ds[1:block_size + 1].unsqueeze(0).repeat(batch_size, 1)
        return x, y
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([ds[i : i + block_size] for i in ix])
    y = torch.stack([ds[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


# --------------------
# Model Configuration
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
d_model = 128
regions_w, regions_h = 8, 8
hexes = make_grid(regions_w, regions_h)
adj = build_adjacency(hexes, long_range_per_node=2)
neighbor_indices = [adj[i] for i in range(len(hexes))]
R = len(hexes)
io_idxs = {"sensor": 0, "motor": R - 1}
print(f"Using {R} regions (grid {regions_w}x{regions_h}), d_model={d_model}")

from cortex.cortex_model import CortexModel
model = CortexModel(
    R=R,
    d_model=d_model,
    neighbor_indices=neighbor_indices,
    io_idxs=io_idxs,
    vocab_size=vocab_size,
    num_facets=7,
    top_m_facets=2,
    k_active=6,
).to(device)

sensor = TextSensor(
    vocab_size=vocab_size, d_model=d_model, ctx_len=block_size, tie_embedding=True
).to(device)
# Tie decoder weights to input embedding
for f in model.mfs.facets:
    f.weight = sensor.emb.weight

# Using the standard Adam optimizer instead of Lion to avoid extra dependencies
optimizer = Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.99), weight_decay=1e-2)

num_steps = 1000
warmup_steps = 50
def lr_lambda(it):
    if it < warmup_steps:
        return float(it + 1) / warmup_steps
    progress = (it - warmup_steps) / max(1, num_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------
# Training Loop
# --------------------
model.train()
print_interval = 10
eval_interval = 200
running_loss = 0.0
print("step,avg_train_loss,val_loss,val_ppl")

for step in range(1, num_steps + 1):
    # anneal k_active from dense -> sparse
    k0, kT, T = R, max(2, R // 8), 2000
    model.gate.k_active = max(kT, int(k0 - (k0 - kT) * min(1.0, step / T)))

    x_batch, y_batch = get_batch("train", batch_size, block_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # Reset state for new sequences
    for region in model.regions:
        region.state.reset_state(batch_size, device)
        region.kv.ptr.zero_()
        region.kv.keys.zero_()
        region.kv.vals.zero_()
    model._neighbor_msg_prev = None

    x_emb, _ = sensor(x_batch)
    losses = []
    for t in range(block_size):
        x_t = x_emb[:, t, :]
        x_per_region = torch.zeros(R, batch_size, d_model, device=device)
        x_per_region[io_idxs["sensor"]] = x_t
        target_t = y_batch[:, t]
        _, loss_t, _ = model(x_per_region, targets=target_t)
        losses.append(loss_t)

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss.item()
    if step % print_interval == 0:
        avg_train_loss = running_loss / print_interval
        running_loss = 0.0
        val_loss = ""
        val_ppl = ""
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_batches = 10
                val_loss_total = 0.0
                for _ in range(val_batches):
                    x_val, y_val = get_batch("val", batch_size, block_size)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    for region in model.regions:
                        region.state.reset_state(batch_size, device)
                        region.kv.ptr.zero_()
                        region.kv.keys.zero_()
                        region.kv.vals.zero_()
                    model._neighbor_msg_prev = None
                    x_emb_val, _ = sensor(x_val)
                    batch_loss = 0.0
                    for t in range(block_size):
                        x_val_t = x_emb_val[:, t, :]
                        x_per_region = torch.zeros(R, batch_size, d_model, device=device)
                        x_per_region[io_idxs["sensor"]] = x_val_t
                        target_val_t = y_val[:, t]
                        _, loss_val_t, _ = model(x_per_region, targets=target_val_t)
                        batch_loss += loss_val_t.item()
                    val_loss_total += batch_loss / block_size
                val_loss = val_loss_total / val_batches
                val_ppl = math.exp(val_loss)
            model.train()
            print(f"{step},{avg_train_loss:.4f},{val_loss:.4f},{val_ppl:.2f}")
        else:
            print(f"{step},{avg_train_loss:.4f},{val_loss},{val_ppl}")


# --------------------
# Text Generation Demo
# --------------------
model.eval()
print("\nGenerating text...")
with torch.no_grad():
    for region in model.regions:
        region.state.reset_state(1, device)
        region.kv.ptr.zero_()
        region.kv.keys.zero_()
        region.kv.vals.zero_()
    model._neighbor_msg_prev = None

    prompt = "The "
    generated = prompt
    for ch in prompt[:-1]:
        idx = torch.tensor([[stoi[ch]]], device=device)
        emb, _ = sensor(idx)
        x_per_region = torch.zeros(R, 1, d_model, device=device)
        x_per_region[io_idxs["sensor"]] = emb[0, 0, :]
        model(x_per_region, targets=None)

    current_char = prompt[-1]
    for _ in range(200):
        idx = torch.tensor([[stoi[current_char]]], device=device)
        emb, _ = sensor(idx)
        x_per_region = torch.zeros(R, 1, d_model, device=device)
        x_per_region[io_idxs["sensor"]] = emb[0, 0, :]
        logp, _, _ = model(x_per_region, targets=None)
        probs = logp.exp()
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_char = itos[next_id]
        generated += next_char
        current_char = next_char

    print(f"Generated text (prompt + 200 chars):\n{generated}")
