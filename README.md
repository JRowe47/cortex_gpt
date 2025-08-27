# nanoGPT‑FE‑RoPE: Cortical columns + geometric routing + MFS head

**Purpose.** This repo explores how to give a tiny GPT a **biologically‑inspired communication structure**—cortical columns grouped into regions with sparse, distance‑aware routing—while keeping the familiar GPT training loop. Concretely, it adds:

* **Fe‑RoPE (“iron RoPE”)** with **anchor‑relative** phases for robust relative geometry, paired with **Poisson‑disk sampled (PDS)** anchors to get **block‑sparse neighbor attention** (“geometric routing”). Implemented inside attention. &#x20;
* **Multi‑Facet Softmax (MFS)** output head to mitigate the **softmax bottleneck** in ambiguous contexts (multi‑modal next‑token distributions). Implemented as an optional head. &#x20;
* A lightweight **sliding‑window Autoencoder (AE)** auxiliary head for **stateful recurrence** (reconstruction + next‑step), encouraging short‑horizon memory without changing the transformer core. Implemented as an optional aux loss.&#x20;
* A **cortical assembly abstraction** (columns → 6 layers → regions) with **sparse‑by‑transmission** signaling and **PDS‑mediated inhibition**. This sits conceptually above the per‑layer attention routing and informs how we wire the model and future modules (e.g., HTM‑style meta‑updates). *Status: design + early scaffolding; core ideas already leveraged by the geometric routing inside attention.*&#x20;

The fork remains drop‑in compatible with nanoGPT training. You can flip features on/off via config flags.

## Installation & Setup

This repo targets **Python 3.10+** and **PyTorch 2.0+**. A minimal setup:

```bash
git clone https://github.com/yourname/cortex_gpt.git  # replace with your fork
cd cortex_gpt
pip install torch numpy tiktoken
```

`tiktoken` is only needed for the sampling helper; the core model depends mainly on PyTorch and NumPy.

---

## Big picture: columns → regions → geometric routing

### Cortical columns (local microcircuits)

* We model a *column* as a **token‑local microcircuit** realized by a standard transformer block (LN → self‑attention → MLP), but we **constrain communication** using **PDS anchors** and **neighbor rings**: tokens that fall into the same anchor cell (or nearby cells) are allowed to talk; others are masked out. That gives us **lateral inhibition** (sparsity) and **short‑range excitation** (local mixing). Implemented as a block mask in attention.&#x20;

### Six cortical layers (functional motif)

* We follow a **6‑layer cortical motif** conceptually:

  * **L4**: fast feedforward feature extraction (**reconstruction**) and **t+1 prediction** (the AE head).
  * **L2/3**: **horizontal pooling** and **contextual mixture** (our cheap “slate” summarization conditions MFS/AE).
  * **L5**: **descending projections** (drives sparse routing outward—mirrors PDS rings between “columns”).
  * **L6**: **feedback/prior** (a slow predictive signal you can later fold into the slate).
  * **L1**: background modulators (not explicitly modelled here).
    These roles are **implemented minimally** via (i) the *geometric* gating of attention (PDS sparsity) and (ii) the **L4 AE head** with a **slate** side channel; additional per‑layer heads are optional extensions.&#x20;

### Regions (meso‑scale topology)

* A **region** is a bundle of columns that mostly talk locally (intra‑region) with a few structured **feedback** edges to itself or other regions (inter‑region). The same **PDS neighbor‑ring rule** scales to the region graph, giving **multi‑scale geometric routing** (columns within regions; regions to regions).
* Today, the *mechanics* of geometric routing are already inside attention (token‑level); region‑level wiring is an **experimental abstraction** built from the same rules and intended to coordinate slate construction and routing masks across layers/blocks.&#x20;

> **TL;DR:** We enforce *where signals are allowed to go* using geometry (PDS anchors + neighbor rings) and *what relative offsets mean* using Fe‑RoPE. That’s the backbone for column/region‑like behavior without introducing heavyweight MoE or explicit graph pipelines. &#x20;

---

## What’s implemented now (and where)

**Transformer core with Fe‑RoPE + PDS sparsity.**
`CausalSelfAttention` computes 1‑D normalized positions, samples PDS anchors, rotates Q/K using **Fourier‑extended RoPE** **relative to the assigned anchor** (iron RoPE), and applies a **block mask** that only keeps **within‑cell** and **neighbor‑ring** interactions. See the attention forward path for `build_pd_anchors_1d`, `anchor_local_offsets`, `ferope_ar_rotate`, and the sparse mask construction.&#x20;
Fe‑RoPE derivation and rationale appear in the Fe‑RoPE notes in this repo.&#x20;

**Multi‑Facet Softmax (MFS) head.**
An optional LM head that mixes **K facets**; the **first softmax** is **partitioned** over **P** disjoint vocab stripes, later softmaxes see the full vocab; a learned prior mixes them. We **autodetect** whether the head returns logits or log‑probs and select **CE vs NLL** loss accordingly. This follows the ACL’22 MFS design; see §3 and Fig. 2 in the paper and the head wiring in `GPT.forward`. &#x20;

**Sliding‑window AE aux head (cheap recurrence).**
An auxiliary head computes **(a)** window **reconstruction** and **(b)** **next‑step** loss (using a left‑shifted `next_targets`, last position ignored), then adds a scalar‑reduced penalty to the main loss. Think of this as **L4 dynamics** encouraging short‑horizon memory. Wired at the tail of `GPT.forward`.&#x20;

**Slate (local multi‑input context).**
A tiny, causal mean‑pool over the left context used to condition the MFS and AE heads (proxy for **L2/3 pooling** and **L6 prior** until the full region slate is added). See `_make_slate`.&#x20;

**Compatibility + fallback.**
Disable MFS to use the standard linear head; disable AE to train pure LM loss; disable PDS to run dense attention with Fe‑RoPE rotations only. The model preserves nanoGPT’s `(logits, loss)` contract. &#x20;

---

## High‑level dataflow (one forward)

1. **Token & position embeddings** → transformer blocks.
2. **Per block attention:**

   * Assign tokens to **PDS anchors**; compute **Δ (token↔anchor)**; apply **Fe‑RoPE** rotations on Q/K using phases `θ=W·Δ`. &#x20;
   * Apply causal mask ∧ **neighbor‑ring** mask → **sparse attention**.&#x20;
3. **Final hidden `h`** →

   * **MFS head** (K facets, P partitions) → logits/log‑probs and LM loss. &#x20;
   * **AE head** (optional): **reconstruction** + **t+1** using `next_targets` → scalar aux loss; mix into LM loss.&#x20;
4. Return **(logits, loss)**; generation uses the logits path as in nanoGPT.&#x20;

---

## Tokenization (T5‑style, byte‑friendly)

This fork is tokenizer‑agnostic (byte‑BPE/BPE/SentencePiece). If you want **T5‑style** segmentation (SentencePiece unigram) or **ByT5** byte‑level robustness, plug it in upstream: Fe‑RoPE, PDS, MFS, and AE are **tokenization‑independent**. Byte‑friendly schemes pair well with **anchor‑relative Fe‑RoPE** because the model leans less on absolute distances and more on **Δ‑geometry**.

---

## Why these pieces exist (intent)

* **Geometric routing** (PDS + Fe‑RoPE) gives a **sparse‑by‑transmission** inductive bias and **translation‑equivariance** in the chosen geometry—more stable long contexts and locality, fewer spurious long‑range attentions. (Design + code.) &#x20;
* **MFS** relaxes the **softmax bottleneck**: when the correct next‑token distribution is **multi‑modal**, a single hidden can’t rank both modes correctly; facets + partitioning help. (Theory + head.) &#x20;
* **Sliding‑window AE** cheaply encourages **statefulness** and **predictive coding** (L4‑like), improving short‑range credit assignment. (Aux head.)&#x20;
* **Columns/regions** unify these into a **multi‑scale** picture: local pooling/inhibition inside columns; structured, sparse inter‑region feedback. (Scaffolded by the same PDS rules already in attention.)&#x20;

---

## Differences vs. vanilla nanoGPT

| Area              | nanoGPT              | This fork                                                        |
| ----------------- | -------------------- | ---------------------------------------------------------------- |
| Positional bias   | Absolute learned WPE | **Fe‑RoPE (iron)**, **anchor‑relative** phases (W·Δ)             |
| Attention         | Dense causal         | **PDS block‑sparse** within/neighbor rings (“geometric routing”) |
| Output head       | Linear → softmax     | **MFS** (K facets, first softmax **partitioned** P stripes)      |
| Aux objectives    | —                    | **Sliding‑window AE** (recon + next‑step)                        |
| Architecture view | Flat stack           | **Columns (6‑layer motif) → regions** (sparse feedback)          |
| Contract          | `(logits, loss)`     | Same, with robust handling of logits vs log‑probs from MFS       |
| Code              | single `model.py`    | Extended `model.py` with Fe‑RoPE+PDS, MFS, AE (feature‑toggled)  |

Baseline nanoGPT `model.py` is included for reference; this fork’s `model.py` integrates all of the above and preserves the API. &#x20;

---

## Configuration (key flags)

```python
GPTConfig(
  # core sizes
  block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True,

  # Fe‑RoPE + PDS (geometric routing)
  ferope_anchor_relative=True, ferope_m=64, rope_base=10000.0,
  use_block_sparse=True, neighbor_rings=1, anchor_min_gap=256, anchor_jitter=32,

  # Heads
  tie_weights=True,
  use_mfs_head=True, mfs_K=3, mfs_P=4, mfs_lowrank_r=0,

  # Auxiliary (L4) losses
  add_l4_losses=True, l4_loss_weight=0.1,
)
```

* **Turn features off**: `use_mfs_head=False` (linear head), `add_l4_losses=False` (no AE), `use_block_sparse=False` (dense attention, Fe‑RoPE kept).&#x20;
* **Fe‑RoPE width**: ensure `2*ferope_m ≤ head_dim`. Defaults mirror the Fe‑RoPE notes.&#x20;

---

## Quickstart

* **Smoke test** (sanity): `python smoke_test.py --device=cpu` runs a tiny model end‑to‑end and prints probability sums ≈ 1 after softmax, a finite train loss, and generation output. The test exercises both **MFS** and fallback head paths.&#x20;
* **Training**: `python train.py --batch_size=32 --compile=False` mirrors the nanoGPT training loop; the model returns `(logits, loss)` and handles CE/NLL automatically based on the MFS head’s output convention.&#x20;
* **Sampling**: once checkpoints exist in `out/`, `python sample.py --out_dir=out --num_samples=5` produces sample completions.&#x20;

---

## Roadmap (biologically‑inspired extras)

* **Region‑level slate & routing.** Promote the current token‑local “slate” to an explicit **region slate** (pool of active columns + L4 error + L6 prior), and export its **PDS mask** so downstream blocks/layers share the same sparse pattern (cross‑layer consistency). (Design principles already reflected in attention routing.)&#x20;
* **HTM‑style online meta‑updates.** Add a light optimizer modulation that **latches onto successful transmissions** (columns receiving consistent prediction successes get a transient LR or momentum boost; inhibited columns decay). Implemented as a wrapper around AdamW; leverages signals already computed by AE/MFS without heavy extra compute. (Concept—scaffold available via the aux losses and slates.)&#x20;
* **Cross‑region feedback.** Wire a few **dense feedback** edges among select regions (motor/sensory‑like loops), still respecting neighbor‑ring constraints at the region graph level.

---

## Files of interest

* **`model.py`** — GPT with Fe‑RoPE (anchor‑relative), **PDS neighbor attention**, **MFS** head, **sliding‑window AE**, slate helper, generation. Feature flags documented in the class.&#x20;
* **Fe‑RoPE notes** — derivation + intuition for multi‑frequency rotary encoding and anchor‑relative geometry (“iron RoPE”).&#x20;
* **Baseline nanoGPT `model.py`** — for reference/contrast.&#x20;
* **MFS paper (ACL 2022)** — theory & results behind multi‑facet softmax; our head mirrors Fig. 2 and §3 (facets + first‑softmax partitioning).&#x20;

---

## FAQs

**Does geometric routing help long context?**
Yes; Fe‑RoPE uses **relative** phases (anchor‑relative) so scores depend on **Δ‑position**, and PDS sparsity curbs spurious long‑range mixing. That reduces OOD failures when contexts grow or slide. (See Fe‑RoPE notes + attention code.) &#x20;

**Why MFS over MoS?**
MFS keeps facets **context‑flexible** (multi‑input slate) and **partitions** the first softmax to break interfering global geometry, addressing classic softmax bottlenecks on ambiguous contexts—with modest compute overhead. (Paper + implementation.) &#x20;

**Do I need a special tokenizer?**
No. Byte‑level (ByT5) and SentencePiece both work; the routing & encoding are tokenization‑independent.

