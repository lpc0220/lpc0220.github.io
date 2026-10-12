---
layout: post
title: "Block-Sparse Attention for Video Diffusion on Blackwell: the Backward Pass"
author: Pengcheng Li
date: 2026-10-12
description: "Blackwell (sm_100a) CUDA kernels for the backward pass of the Video Sparse Attention fine stage: design and measured performance."
---

## 1. Background: the VSA backward

The [first post]({% post_url 2026-10-12-vsa-blackwell %}) covers the VSA fine stage, in which each query block attends only to its own top-k KV blocks. This post covers its backward pass.

Given the output gradient dO, the backward recomputes the attention probabilities from the log-sum-exp (LSE) saved by the forward pass, so it needs no softmax reduction:

- P = exp2(S · scale · log2 e − LSE), with S = Q·Kᵀ
- dV = Pᵀ·dO
- dP = dO·Vᵀ, and dS = P ∘ (dP − Δ), with Δ = rowsum(dO ∘ O)
- dQ = dS·K · scale, and dK = dSᵀ·Q · scale

That is five GEMMs per (query block, KV block) pair, against two in the forward pass, so the backward does 2.5× the FLOPs.

Sparsity adds a second problem (Figure 1). The forward pass walks each query block's top-k list: a row of the mask. dQ of a query block is a sum over the same row, but dK and dV of a KV block are sums over a column: the query blocks that selected it. Every row has exactly top-k entries; column lengths vary, and a KV block that no query block selected still needs zero dK and dV. An inverted index gives each KV block its list of query blocks.

![The mask read by row and by column](/assets/images/vsa-blackwell-backward/bwd-lists.svg)

*Figure 1. The forward pass reads the mask by row; the backward also reads it by column.*

## 2. Kernel design

### 2.1 Starting point: one KV block per CTA, three kernels

The main kernel is KV-stationary. One CTA owns one KV block: it loads K and V once and walks the block's inverted list, one query tile per step. dK and dV accumulate in TMEM over the whole list and are stored once at the end. dQ is the hard output. Each step produces a partial dQ for the step's query tile, summed over this CTA's keys only; the full dQ of a query block is the sum of the partials from every KV block it selected. The CTA adds its partial into an fp32 accumulator in global memory with a bulk TMA reduce-add (`cp.reduce.async.bulk`).

The alternative is two kernels: one KV-stationary for dK and dV, one query-stationary for dQ. That avoids the global accumulation, but both kernels recompute S and dP: seven GEMMs per pair instead of five. Our Triton backward takes that route.

The main kernel sits between two small kernels:

- **Preprocess:** computes Δ = rowsum(dO ∘ O) per query row and zeroes the dQ accumulator.
- **Postprocess:** multiplies the accumulator by the softmax scale and stores dQ in bf16.

The three are chained with programmatic dependent launch (PDL). Each kernel launches and sets up (barrier init, TMEM allocation) while the previous one drains, and waits (`griddepcontrol.wait`) only right before it reads the previous kernel's output. This removes a constant 4–10 µs per call: 5–8% at 4K tokens, nothing measurable from 32K up.

### 2.2 128-token blocks: one KV block per CTA, one query block per step

A 128-token KV block is exactly one 128-row MMA tile, and each step takes one 128-token query block from its list (Figure 2). All five GEMMs are plain 128 × 128 `tcgen05` MMAs with the KV block as the M dimension:

1. Sᵀ = K·Qᵀ
2. dPᵀ = V·dOᵀ
3. dV += Pᵀ·dO, with Pᵀ read from TMEM
4. dQ = dS·K, with dS read from shared memory
5. dK += dSᵀ·Q, with dSᵀ read from TMEM

**TMEM.** The 512 columns hold four 128-column regions:

- 0–127: Sᵀ, then Pᵀ in bf16 written over it.
- 128–255: the dV accumulator.
- 256–383: dPᵀ, then dSᵀ in bf16 over it, then the step's dQ tile.
- 384–511: the dK accumulator.

Regions are reused within a step, so the MMA warp issues in an order that lets the next step's recompute run ahead: Sᵀ(j+1), dK(j), dQ(j), dPᵀ(j+1), dV(j+1). dPᵀ(j+1) waits until the dQ warps have read dQ(j) out of columns 256–383. dK(j) reads dSᵀ from those same columns before dQ(j) overwrites them; no barrier is needed, because the tensor core executes MMAs in issue order.

**Warps.** Each CTA runs 16 warps:

- **8 compute warps** form Pᵀ = exp2(Sᵀ · scale − LSE) and dSᵀ = Pᵀ ∘ (dPᵀ − Δ). A warp can access only the 32 TMEM lanes given by its warp ID mod 4, so the eight split each tile into four 32-lane groups × two 64-column halves; a thread handles one KV row × 64 query columns. They write dSᵀ twice from registers: bf16 into TMEM for GEMM 5, and into shared memory for GEMM 4, which reads it transposed.
- **4 dQ warps** read the whole dQ tile out of TMEM, one query row per thread, which frees the columns for the next dPᵀ. They then push it to the accumulator in four 16 KB chunks of 32 head-dim columns, through two shared-memory buffers.
- **1 MMA warp, 1 TMA load warp**, and 2 idle warps that donate their registers.

`setmaxnreg` rebalances registers per role: 152 per thread for the dQ warps, which hold a 128-column dQ row, 136 for the compute warps, 88 for MMA and load, 24 for the idle warps.

**One copy of Q and dO.** Q is read by two GEMMs that contract over different dimensions: Sᵀ = K·Qᵀ over the head dimension, dK += dSᵀ·Q over tokens. The kernel keeps a single copy of each Q tile in shared memory, in its natural [token, head-dim] layout, and the second GEMM reads it through the instruction descriptor's transpose bit. The same holds for dO. Q is double-buffered and dO single-buffered; K and V ride the first step's Q and dO loads as extra TMA bytes.

The grid has one CTA per (KV block, head). A KV block with an empty list takes a short path that stores zero dK and dV.

![128-token backward: shared memory, the five GEMMs and the TMEM map](/assets/images/vsa-blackwell-backward/bwd-blk128.svg)

*Figure 2. 128-token blocks: K and V stay in shared memory and dK and dV in TMEM for the whole list; only the dQ partial leaves the SM each step.*

### 2.3 The dQ bottleneck

Each step of the 128-token kernel adds a 128 × 128 fp32 dQ tile (64 KB) into the accumulator: a read-modify-write in L2. That traffic grows with S² · density, exactly like the FLOPs, so a longer sequence never amortizes it. We measured the bulk reduce-add at about 48 GB/s per SM, 6.5 TB/s across the GPU. At 128K tokens, profiling shows the dQ stream limited by latency (54% L2 hit rate) rather than DRAM bandwidth (40% utilized).

Smaller blocks make it worse. A 64-token KV block pushes a 32 KB partial per (64-row query block, KV block) pair, which carries a quarter of the FLOPs of a 128-token pair: twice the reduce-add bytes per FLOP. Sections 2.4–2.6 handle this per block size.

### 2.4 64-token blocks: one KV block per CTA, four query blocks per step

A 64-row KV block fills half of a 128-row MMA. As in the forward pass, `tcgen05.mma.ws` (Layout E) runs two independent 64-row products in one 128-lane MMA (Figure 3). The CTA walks its KV block's list in quads of four query blocks, 256 query tokens per step. Sᵀ of a quad is 64 KV rows × 256 query columns, stored in two halves: lanes 0–63 hold query blocks 0 and 2, lanes 64–127 hold blocks 1 and 3. dV and dK accumulate the same way, as two partial sums in the two lane halves, which are added together before the final store. dQ uses two plain 128-row MMAs per quad, one per lane half.

When the list length is not a multiple of four, the last quad repeats the list's last block, and the compute warps set P and dS to zero at the repeated positions, so they add nothing.

Each quad gathers 256 tokens of Q and dO, and each is read by two GEMMs. They go through one shared-memory ring of four 32 KB slots: dOᵀ for blocks 0–1 and 2–3, Qᵀ for blocks 0–1 and 2–3. A slot is freed by the commit of its last-reading GEMM. dV frees the dO slots before dK frees the Q slots, so the loader fills the next quad's dO first.

The single-copy rule holds here too, in the other direction: the preprocess also writes Qᵀ and dOᵀ ([head-dim, tokens]), the layout the packed dV and dK GEMMs need, and Sᵀ and dPᵀ read that same copy through the transpose bit. Loading both layouts made the load warp the bottleneck; one copy measured 4–31% faster.

![64-token backward: quads, the ring and the dual-packed Sᵀ](/assets/images/vsa-blackwell-backward/bwd-blk64.svg)

*Figure 3. 64-token blocks: four query blocks per step, packed into the two lane halves of every MMA.*

### 2.5 64-token blocks: two passes

With fp32 dQ, the one-pass 64-token kernel levels off at 720–810 TFLOPS from 32K tokens up, held there by the reduce-add stream. The two-pass variant removes the read-modify-write altogether (Figure 4):

- **Pass 1** is the KV-stationary kernel above without the dQ path. It computes dK and dV and stores each 64 × 64 dSᵀ tile (8 KB, bf16) to a global buffer, at the tile's query block and its position in that block's top-k list.
- **Pass 2** is query-stationary. Each work item is one query block and walks the block's own top-k list; each step loads the stored dS tile and gathers the KV block's K. dQ accumulates in TMEM over the whole list and is stored once in bf16: no accumulator, no postprocess, and no recomputation, since dS is already known.

The price is the dS buffer: B · H · blocks · top-k · 8 KB, 68.7 GB at 128K tokens with 8 heads. At long sequences the passes run one head at a time, so the buffer holds one head. Two passes are 5–18% faster than one pass up to 128K tokens. At 256K they are equal, and at 512K two passes are 15% slower, because the per-head slabs no longer fit in L2. The dispatch uses two passes up to 128K tokens and one pass from 256K.

![One pass versus two passes for 64-token blocks](/assets/images/vsa-blackwell-backward/bwd-twopass.svg)

*Figure 4. One pass pushes an fp32 dQ partial per pair; two passes store bf16 dS and sum dQ per query block on chip.*

### 2.6 256-token blocks: two CTAs per KV block

A 256-token KV block is two 128-row MMA tiles. In the 1-CTA form, two independent CTAs each own one 128-row half and run the 128-token kernel's loop over the block's list of 128-row query tiles. Each loads every Q and dO tile in full, and each pushes its own dQ partial: two 64 KB reduce-adds per query tile.

The 2-CTA form pairs the two CTAs with `cta_group::2` (Figure 5). The pair issues joint MMAs with M = 256 over the whole KV block, each CTA keeping its 128 rows in its own TMEM. Each CTA loads half of every Q and dO tile, in both layouts the joint MMAs need. dQ = dS·K contracts over all 256 keys, so the CTAs exchange their dS halves through distributed shared memory and compute dQ as one cluster GEMM; each CTA then pushes its own 64 query rows. That is one reduce-add per query tile instead of two, which halves the dQ traffic per FLOP. Q and dO bytes per CTA stay the same.

Unlike in the forward pass, the pair pays here, because the stationary side is the KV block: the 256 rows of one KV block share one list by construction. The 2-CTA form is 5–13% faster than the 1-CTA form from 4K to 512K tokens, with the largest gain at 512K.

At 128-token blocks the same pairing splits one KV block into two 64-row halves. It measured 4–21% slower than 1-CTA. Tensor-core time per tile is unchanged, so the likely cause is per-step overhead, such as the cross-CTA barriers and the dS exchange, now spread over half the work.

![1-CTA and 2-CTA forms for 256-token blocks](/assets/images/vsa-blackwell-backward/bwd-blk256.svg)

*Figure 5. 256-token blocks: the 2-CTA pair contracts dQ over all 256 keys before pushing it, halving the reduce-add traffic.*

### 2.7 Summary

| Block size | KV rows per CTA | Query rows per step | MMA form | dQ path |
| --- | ---: | ---: | --- | --- |
| 64 | 64 | 256 (4 blocks) | two 64-row products per MMA | two passes up to 128K tokens; fp32 reduce-add from 256K |
| 128 | 128 | 128 | M = 128 | fp32 reduce-add, one per pair |
| 256 | 128 (2-CTA pair) | 128 | M = 256 across the pair | fp32 reduce-add, one per query tile per 256 keys |

All three keep K and V in shared memory and dK and dV in TMEM for the whole list. They differ in how much dQ traffic each FLOP costs.

## 3. Performance improvement

All kernel-level results use one setup:

| Component | Setup |
| --- | --- |
| GPU | NVIDIA GB200 (sm_100a), CUDA 13.0 |
| Shape | B=1, H=8, head dim 128, bf16, non-causal MHA |
| Sparsity | 25% density (top-k = number of blocks / 4), random block selection shared across heads |
| Inputs | identical saved tensors for every implementation |
| Metric | TFLOPS over selected block pairs: 2.5 × 4·D·H·blocks·top-k·block² |

At a fixed sequence length and density, every block size does the same number of FLOPs, so the TFLOPS of different block sizes are directly comparable. The baselines are our Triton backward at 64 tokens and FA4's backward at 128 and 256 tokens (its best configuration per size). Ours were measured in one session; the baselines in separate sessions on the same node and inputs, with ±3% session-to-session spread. Our times include the preprocess and postprocess kernels.

![TFLOPS vs sequence length](/assets/images/vsa-blackwell-backward/bwd-perf.svg)

*Figure 6. Backward throughput on GB200.*

| Block | Baseline | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Triton | 2.35× | 2.20× | 2.21× | 2.12× | 1.91× | 1.81× | 1.70× | 1.73× |
| 128 | FA4 | 2.25× | 1.47× | 1.13× | 1.06× | 1.02× | 1.03× | 1.03× | 1.03× |
| 256 | FA4 | 3.72× | 1.89× | 1.28× | 1.07× | 1.00× | 0.97× | 0.98× | 0.98× |

The largest gains are at short sequences. From 64K tokens on, the 128- and 256-token results are within the ±3% session spread of FA4: parity, not a win. Our kernels and FA4 slow down past 32K tokens; for ours, the dQ reduction stream (Section 2.3) sets the pace there. Every kernel is checked against an fp32 reference of the full dQ, dK and dV.

## Acknowledgements

Thanks to the FastVideo maintainers for reviewing and merging the kernels.
