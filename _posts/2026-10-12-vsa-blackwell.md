---
layout: post
title: "Block-Sparse Attention for Video Diffusion on Blackwell: VSA Forward and Backward Kernels"
author: Pengcheng Li
date: 2026-10-12
description: "Blackwell (sm_100a) CUDA kernels for the fine stage of Video Sparse Attention: design and measured performance of the forward and backward passes."
---

## 1. What the kernels compute

[VSA](https://arxiv.org/abs/2505.13389) (Zhang et al., 2025) replaces full attention in video diffusion transformers with two stages. A coarse stage attends over mean-pooled token blocks and keeps the top-k KV blocks for each query block. A fine stage then computes exact attention, but only over the selected blocks (Figure 1). The fine stage dominates the cost, and it is the part these kernels implement.

![VSA fine stage as a block-sparse mask](/assets/images/vsa-blackwell/vsa-fine-stage.svg)

*Figure 1. The fine stage is block-sparse attention: each query block attends to its own top-k KV blocks.*

The problem differs from dense attention in three ways. Every query block has its **own** KV list, so K/V must be gathered rather than streamed. The block size is set by the model (FastVideo's default is a 4×4×4 = 64-token video cube), and 64 rows is half the height of a Blackwell tensor-core tile. And the backward pass has to route gradients through that irregular pattern.

## 2. Forward kernel

**Start from dense attention.** The forward kernel is our dense Blackwell attention kernel, which follows the FlashAttention-4 forward design, with only the sparse parts changed. Each CTA runs 16 specialized warps: a TMA load warp, a `tcgen05` MMA warp, eight softmax warps, four correction warps, an epilogue warp, and a scheduler warp that steals work with cluster launch control (CLC). Each CTA processes two 128-row M-tiles whose S and O accumulators fill the 512 tensor-memory (TMEM) columns. Sparsity changes two things: the load warp gathers each selected KV block by its index instead of streaming a contiguous range, and the K loop visits only the selected blocks.

**128-token blocks: the dense tiling.** A 128-token query block is exactly one 128-row M-tile, and a 128-token KV block is exactly one 128-key K/V tile, so the MMAs, TMEM layout and softmax are the dense kernel's (Figure 2). Each CTA takes two adjacent query blocks, one per M-tile, each with its own list. At each step, query block 2p computes S = Q·Kᵀ against the next block in its list: a 128 × 128 tile with one TMEM lane per row. Four softmax warps turn S into P, one thread per row, and write P back over S. P·V then accumulates into O. Query block 2p+1 does the same in its own TMEM columns. The MMA warp alternates between the two blocks, so one block's softmax overlaps the other's MMAs.

The one structural difference is K/V reuse. Dense attention loads each K/V tile once and feeds it to both M-tiles. Here the two query blocks have different lists, so the load warp gathers a separate K and V tile for each, twice the loads per step, and interleaves them in one shared-memory ring in the order the MMA warp consumes them.

![128-token blocks: dense tiling with two gathered K/V streams](/assets/images/vsa-blackwell/fwd-blk128.svg)

*Figure 2. 128-token blocks: each query block is one M-tile with its own list; S and O use the dense TMEM layout, but each M-tile needs its own K/V tiles.*

**64-token blocks: two query blocks per CTA, four KV blocks per step.** A 64-row query block fills only half of a 128-row MMA. Each CTA again takes two adjacent query blocks, each with its own list, and interleaves their K/V in one ring. TMEM is 128 lanes (rows) × 512 columns, and each query block gets 128 columns for its scores. Each step of the K loop gathers four selected KV blocks (256 keys), so the scores form a 64 × 256 tile, twice as wide as the 128 columns available. `tcgen05.mma.ws` (PTX data-path Layout E) stores it in two halves: scores for keys 0–127 go to lanes 0–63, and scores for keys 128–255 go to lanes 64–127 (Figure 3). Every lane is used, and each query row appears once in each half. Each softmax thread runs an online softmax over one half (128 keys), and the correction warps merge the two halves of each row.

![Dual-packed S in TMEM for 64-token blocks](/assets/images/vsa-blackwell/fwd-dualpack.svg)

*Figure 3. 64-token blocks: each query block gathers four KV blocks from its own list per step; its 64 × 256 score tile uses all 128 TMEM lanes, with keys 0–127 in lanes 0–63 and keys 128–255 in lanes 64–127.*

**256-token blocks: one query block per CTA, one shared K/V stream** (Figure 4).

- **Rows.** A 256-token query block has 256 rows, enough for both M-tiles. Each CTA takes one query block: rows 0–127 are M-tile 0 and rows 128–255 are M-tile 1. Both M-tiles belong to the same block, so they share one top-k list.
- **Keys.** A step still covers 128 keys, as with 128-token blocks. TMEM has 512 columns, and each M-tile needs one S column per key plus 128 O columns, so 2 × (128 + 128) = 512 is the limit. A selected 256-token KV block therefore takes two steps: keys 0–127, then keys 128–255. Each step loads one 128-key K tile and one V tile, with one TMA each.
- **Reuse.** Both M-tiles read the same keys, so each K/V tile is loaded once, and both M-tiles' MMAs read it from the same shared-memory slot, as in dense attention. That is two loads per step, versus four for 128-token blocks.

The TMEM layout and softmax are the same as for 128-token blocks. Our first 256-token kernel still loaded a separate copy of every tile for each M-tile, as the 128-token kernel does, and trailed FA4 at 128K tokens (0.94×). Sharing the stream made it faster than FA4 at every length. It is also 44% faster than our 64-token kernel at 4K tokens and 32% at 64K, from fewer, longer gathers and fewer index lookups.

![256-token blocks: one query block per CTA and one shared K/V stream](/assets/images/vsa-blackwell/fwd-blk256.svg)

*Figure 4. 256-token blocks: the CTA's two M-tiles are the two halves of one query block, so every 128-key K/V tile is loaded once and read by both.*

**512-token blocks: why we pair two CTAs here, and only here.** Blackwell's `cta_group::2` MMA lets two CTAs compute one 256-row tile together. We use it for 512-token blocks only. At 128 and 256 tokens we built 2-CTA kernels too, and they were slower than 1-CTA.

**Why pair CTAs at all.** A loaded K/V tile can be used only by rows whose accumulators are in the CTA's TMEM at that moment. Each 128-row M-tile keeps S (128 columns) and O (128 columns) there, and O stays for the whole K loop, so one CTA's 512 columns hold two M-tiles: 256 rows. A pair of CTAs holds four M-tiles, 512 rows, and one K/V tile can feed all of them.

**Why only 512-token blocks.** Pairing pays only if all 512 rows of the pair share one list, so that every K/V tile the pair loads is used by all of them. Only 512-token blocks give that (Figure 5):

| Block size | The pair's 512 rows are | Lists | What the pair loads | 2-CTA vs 1-CTA | Use 2-CTA? |
| --- | --- | ---: | --- | --- | --- |
| 128 | 4 query blocks | 4 | each joint MMA covers two blocks, so it loads the union of their lists: 2.7× more KV blocks at 25% density | 0.4–0.5× | no |
| 256 | 2 query blocks | 2 | two separate K/V streams; each CTA loads the same bytes as in 1-CTA | 0.91–0.97× | no |
| 512 | 1 query block | 1 | one K/V stream; each CTA loads half of every tile | fastest at 8K and 128K, within 3% elsewhere | yes |

**How the 512-token kernel splits the work.** CTA 0 holds rows 0–127 and 256–383 of the query block, and CTA 1 holds rows 128–255 and 384–511. A step is still 128 keys, so a selected 512-token KV block takes four steps. In each step, CTA 0 loads keys 0–63 of the K tile and CTA 1 loads keys 64–127; V is split the same way along the head dimension. Each joint MMA reads both halves, so K/V loads per row halve: L2 read requests drop 1.95× at 32K tokens. The speedup is small because DRAM traffic doesn't change (3–4% of HBM bandwidth): the saved requests were L2 hits, and the kernel is compute-bound.

![2-CTA geometry for 128-, 256- and 512-token blocks](/assets/images/vsa-blackwell/fwd-2cta.svg)

*Figure 5. A 2-CTA cluster covers 512 query rows. Colors mark distinct top-k lists.*

**Summary.** Every kernel runs two 128-lane M-tiles per CTA and ping-pongs between them: while the softmax warps work on one tile, the MMA warp runs the other tile's MMAs. TMEM fits exactly two (2 × (128 S + 128 O) = 512 columns). The block size decides what fills the two tiles and whether they can share K/V:

| Block size | Query blocks per CTA | Rows per M-tile | KV blocks per step | Top-k lists per CTA | M-tiles share K/V? |
| --- | ---: | ---: | ---: | ---: | --- |
| 64 | 2 | 64 | 4 | 2 | no |
| 128 | 2 | 128 | 1 | 2 | no |
| 256 | 1 | 128 | ½ | 1 | yes |
| 512 | ½ | 128 | ¼ | 1 | yes, across both CTAs |

KV blocks per step count one M-tile. Small blocks lose reuse because the two M-tiles hold different query blocks with different lists. From 256 tokens on, both tiles belong to one block and reuse matches dense attention, and the 512-token 2-CTA kernel doubles it.

**FastVideo's extensions cost nothing measurable.** The upstream kernel also supports per-block valid lengths (padded video edges), returns the log-sum-exp for the backward pass, and reads V in its natural layout. A full-block check skips the key mask whenever a block is complete. With all three added, throughput moved by +0.7% (64-token blocks) and −0.4% (128-token blocks), within run-to-run noise.

## 3. Backward kernel

**Structure.** The backward pass runs three kernels, chained with programmatic dependent launch (PDL) so that each kernel's launch and prologue overlap the previous kernel's tail (Figure 6). The main kernel is KV-stationary: one CTA owns one KV block, loads K and V once, and walks the list of query blocks that selected it (FastVideo's inverted index). It accumulates dK and dV in TMEM and sends each dQ partial to an fp32 global accumulator with a TMA reduce-add. The tiles, warp roles and TMEM map follow the FA4 sm100 backward, and we deviate only where sparsity forces it.

![Backward pipeline](/assets/images/vsa-blackwell/bwd-pipeline.svg)

*Figure 6. Backward pipeline: preprocess, KV-stationary main kernel, postprocess.*

**dQ is the bottleneck.** dK and dV stay on chip, but a query block's dQ is a sum over its top-k KV blocks, which live in different CTAs. That sum goes through L2 as fp32 read-modify-write traffic. The traffic grows with S²·density, the same as the FLOPs. At 64-token blocks it caps the one-pass kernel at about 750 TFLOPS no matter how fast the math is. Instead of lowering the accumulator's precision, the 64-token backward has a **two-pass** variant:

- **Pass 1** computes dK and dV and stores each 64×64 dS tile (8 KB, bf16).
- **Pass 2** gathers the stored tiles per query block and computes dQ with no read-modify-write at all.

It is 5–18% faster than one-pass up to 128K tokens. At 256K the two are equal, and at 512K two-pass is 15% slower, so we dispatch to one-pass from 256K tokens.

**One copy of Q and dO in shared memory.** The backward pass reads Q in two GEMMs that contract over different dimensions (Sᵀ = K·Qᵀ over the head dimension, dK = dSᵀ·Q over tokens), and dO likewise. Loading both layouts made the load warp the bottleneck. We keep a single transposed copy and let the first GEMM read it through the instruction descriptor's transpose bit, which measured 4–31% faster than loading both layouts.

**Small and launch-bound cases.** PDL removes a constant 4–10 µs per call, worth 5–8% at 4K tokens and nothing measurable from 32K up. At 256-token blocks, a 2-CTA pair covers one KV block with joint 256-row MMAs, and each CTA loads only its half of every Q and dO tile. Unlike in the forward pass, this pays here: +6% at 4K and +13% at 512K over 1-CTA.

## 4. Experiments

All kernel-level results use one setup:

| Component | Setup |
| --- | --- |
| GPU | NVIDIA GB200 (sm_100a), CUDA 13.0 |
| Shape | B=1, H=8, head dim 128, bf16, non-causal MHA |
| Sparsity | 25% density (top-k = number of blocks / 4), random block selection shared across heads |
| Inputs | identical saved tensors for every implementation |
| Metric | TFLOPS over selected block pairs: 4·D·H·blocks·top-k·block² per forward, 2.5× that per backward |

At a fixed sequence length and density, every block size does the same number of FLOPs, so the TFLOPS of different block sizes are directly comparable. Each kernel is compared with the strongest public kernel at the same block size.

### 4.1 Forward

The baselines are FastVideo's Triton kernel and FlashInfer at 64 tokens, FlashInfer at 128, and FA4's block-sparse kernel (FastVideo's VSA-256 path) at 256, which is FA4's minimum block size. Each comparison ran in one session on identical inputs.

![Forward TFLOPS vs sequence length](/assets/images/vsa-blackwell/fwd-perf.svg)

*Figure 7. Forward throughput on GB200.*

| Block | Baseline | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Triton | 3.34× | 2.74× | 2.34× | 2.25× | 2.14× | 1.92× |
| 64 | FlashInfer | 8.78× | 4.65× | 2.67× | 1.76× | 1.32× | 1.14× |
| 128 | FlashInfer | 4.21× | 2.09× | 1.26× | 1.18× | 1.13× | 1.09× |
| 256 | FA4 | 4.13× | 2.17× | 1.33× | 1.16× | 1.07× | 1.04× |

The largest gains are at short sequences, where fixed per-launch and per-tile overheads dominate the baselines.

### 4.2 Backward

The baselines are FastVideo's Triton backward at 64 tokens, and FA4's sm100 backward at 128 and 256 tokens (its best configuration per size). Ours were measured in one session; the baselines in separate sessions on the same node and inputs, with ±3% session-to-session spread. Times include preprocess and postprocess.

![Backward TFLOPS vs sequence length](/assets/images/vsa-blackwell/bwd-perf.svg)

*Figure 8. Backward throughput on GB200.*

| Block | Baseline | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Triton | 2.35× | 2.20× | 2.21× | 2.12× | 1.91× | 1.81× | 1.70× | 1.73× |
| 128 | FA4 | 2.25× | 1.47× | 1.13× | 1.06× | 1.02× | 1.03× | 1.03× | 1.03× |
| 256 | FA4 | 3.72× | 1.89× | 1.28× | 1.07× | 1.00× | 0.97× | 0.98× | 0.98× |

From 64K tokens on, the 128- and 256-token results are within the ±3% session spread of FA4: parity, not a win. Every implementation slows down past 32K tokens. For ours, profiling at 128K points to latency in the dQ reduction stream (54% L2 hit rate) rather than DRAM bandwidth (40% utilized).

### 4.3 In FastVideo

The 64- and 128-token kernels are upstream in FastVideo: the forward in [#1719](https://github.com/hao-ai-lab/FastVideo/pull/1719), the backward in [#1819](https://github.com/hao-ai-lab/FastVideo/pull/1819) and [#1866](https://github.com/hao-ai-lab/FastVideo/pull/1866). With `FASTVIDEO_VSA_SM100A=1`, FastVideo routes the VSA fine stage to these kernels when the inputs are supported (bf16, head dim 128, 64- or 128-token blocks). Unsupported inputs fall back to Triton. Measured through `block_sparse_attn_from_indices` and `torch.autograd.grad`, with 40 heads and Wan-sized sequence lengths:

| Pass | Shape (padded S, blocks, top-k) | Triton | This work | Speedup |
| --- | --- | ---: | ---: | ---: |
| Forward | 480P: 39,936, 624, 62 | 6.47 ms | 3.08 ms | 2.10× |
| Forward | 720P: 92,160, 1,440, 144 | 33.36 ms | 16.26 ms | 2.05× |
| Backward | 480P-like: 39,936, 624, 156 | 47.20 ms | 27.66 ms | 1.71× |
| Backward | 720P-like: 92,160, 1,440, 360 | 247.66 ms | 149.24 ms | 1.66× |

The forward rows use 90% sparsity (top-k ≈ 10% of blocks); the backward rows use 25% density and include FastVideo's index inversion in both columns. Both backwards are validated against an fp32 reference; at 128-token blocks the CUDA and Triton gradients agree to a relative error of at most 2×10⁻².

## 5. Limitations and next steps

- All numbers are from GB200. The kernels also build for GB300 (sm_103a, [#1833](https://github.com/hao-ai-lab/FastVideo/pull/1833)), but most of the tuning was done on GB200.
- FastVideo routes only the 64- and 128-token kernels; its 256-token path uses FA4.
- Past 32K tokens the backward is limited by the fp32 dQ reduction stream. A work order that keeps each query block's partials L2-resident is the next step.
- Vera Rubin (sm_107a) support is planned.

## Acknowledgements

Thanks to the FastVideo maintainers for reviewing and merging the kernels, and to the FlashAttention-4 authors, whose design both kernels follow.
