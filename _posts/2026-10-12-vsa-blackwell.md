---
layout: post
title: "Block-Sparse Attention for Video Diffusion on Blackwell"
author: Pengcheng Li
date: 2026-10-12
description: "Blackwell (sm_100a) CUDA kernels for the fine stage of Video Sparse Attention: design and measured performance."
---

## 1. Background: the VSA fine stage

Our [VSA paper](https://arxiv.org/abs/2505.13389) replaces full attention in video diffusion transformers with two stages. A coarse stage attends over mean-pooled token blocks and keeps the top-k KV blocks for each query block. A fine stage then computes exact attention, but only over the selected blocks (Figure 1). The fine stage dominates the cost, and it is the part these kernels implement.

![VSA fine stage as a block-sparse mask](/assets/images/vsa-blackwell/vsa-fine-stage.svg)

*Figure 1. The fine stage is block-sparse attention: each query block attends to its own top-k KV blocks.*

The problem differs from dense attention in two ways. Every query block has its **own** KV list, so K/V must be gathered rather than streamed. And the block size is set by the model (our default is a 4×4×4 = 64-token video cube), and 64 rows is half the height of a Blackwell tensor-core tile.

## 2. Kernel design

### 2.1 Starting point: dense attention

The kernel is our dense Blackwell attention kernel with only the sparse parts changed. Each CTA runs 16 specialized warps: a TMA load warp, a `tcgen05` MMA warp, eight softmax warps, four correction warps, an epilogue warp, and a scheduler warp that steals work with cluster launch control (CLC). Each CTA processes two 128-row M-tiles whose S and O accumulators fill the 512 tensor-memory (TMEM) columns.

**Ping-pong between the two M-tiles.** Each step of the K loop has two phases that run on different hardware. The MMAs (S = Q·Kᵀ and O += P·V) run on the tensor cores. The softmax (row max, exp2, row sum, and writing P) runs on the SM's general-purpose and special-function units. With one M-tile, each phase would wait for the other, and the tensor cores would sit idle during every softmax. With two, the MMA warp alternates: it issues tile 0's P·V and next Q·Kᵀ while the softmax warps work on tile 1's scores, then switches. Each unit is always busy with one of the tiles.

**Eight softmax warps, four per M-tile.** A warp can access only the 32 TMEM lanes that match its position in its warpgroup (warp ID mod 4). Covering a tile's 128 lanes therefore takes four warps, one thread per lane, which is one thread per query row. Each thread loads its row's 128 scores into registers and computes the row max, exp2 and row sum on its own, with no communication between threads. It then writes P back over S in TMEM. A separate group of four warps per tile lets both tiles' softmax run at the same time, so neither tile waits for the other's. The four correction warps rescale O when a row's running max grows and normalize it at the end, which keeps that work off the softmax's path.

Sparsity changes two things: the load warp gathers each selected KV block by its index instead of streaming a contiguous range, and the K loop visits only the selected blocks.

### 2.2 128-token blocks: two query blocks per CTA, one KV block per step

A 128-token query block is exactly one 128-row M-tile, and a 128-token KV block is exactly one 128-key K/V tile, so the MMAs, TMEM layout and softmax are the dense kernel's (Figure 2). Each CTA takes two adjacent query blocks, one per M-tile, each with its own list. At each step, query block 2p computes S = Q·Kᵀ against the next block in its list: a 128 × 128 tile with one TMEM lane per row. Four softmax warps turn S into P, one thread per row, and write P back over S. P·V then accumulates into O. Query block 2p+1 does the same in its own TMEM columns. The MMA warp alternates between the two blocks, so one block's softmax overlaps the other's MMAs.

The one structural difference is K/V reuse. Dense attention loads each K/V tile once and feeds it to both M-tiles. Here the two query blocks have different lists, so the load warp gathers a separate K and V tile for each, twice the loads per step, and interleaves them in one shared-memory ring in the order the MMA warp consumes them.

![128-token blocks: dense tiling with two gathered K/V streams](/assets/images/vsa-blackwell/fwd-blk128.svg)

*Figure 2. 128-token blocks: each query block is one M-tile with its own list; S and O use the dense TMEM layout, but each M-tile needs its own K/V tiles.*

### 2.3 64-token blocks: two query blocks per CTA, four KV blocks per step

A 64-row query block fills only half of a 128-row MMA. Each CTA again takes two adjacent query blocks, each with its own list, and interleaves their K/V in one ring. TMEM is 128 lanes (rows) × 512 columns, and each query block gets 128 columns for its scores. Each step of the K loop gathers four selected KV blocks (256 keys), so the scores form a 64 × 256 tile, twice as wide as the 128 columns available. `tcgen05.mma.ws` (PTX data-path Layout E) stores it in two halves: scores for keys 0–127 go to lanes 0–63, and scores for keys 128–255 go to lanes 64–127 (Figure 3). Every lane is used, and each query row appears once in each half. Each softmax thread runs an online softmax over one half (128 keys), and the correction warps merge the two halves of each row.

![Dual-packed S in TMEM for 64-token blocks](/assets/images/vsa-blackwell/fwd-dualpack.svg)

*Figure 3. 64-token blocks: each query block gathers four KV blocks from its own list per step; its 64 × 256 score tile uses all 128 TMEM lanes, with keys 0–127 in lanes 0–63 and keys 128–255 in lanes 64–127.*

### 2.4 256-token blocks: one query block per CTA, half a KV block per step

The CTA's two M-tiles are the two halves of one query block, so they share one K/V stream (Figure 4).

- **Rows.** A 256-token query block has 256 rows, enough for both M-tiles. Each CTA takes one query block: rows 0–127 are M-tile 0 and rows 128–255 are M-tile 1. Both M-tiles belong to the same block, so they share one top-k list.
- **Keys.** A step still covers 128 keys, as with 128-token blocks. TMEM has 512 columns, and each M-tile needs one S column per key plus 128 O columns, so 2 × (128 + 128) = 512 is the limit. A selected 256-token KV block therefore takes two steps: keys 0–127, then keys 128–255. Each step loads one 128-key K tile and one V tile, with one TMA each.
- **Reuse.** Both M-tiles read the same keys, so each K/V tile is loaded once, and both M-tiles' MMAs read it from the same shared-memory slot, as in dense attention. That is two loads per step, versus four for 128-token blocks.

The TMEM layout and softmax are the same as for 128-token blocks. Our first 256-token kernel still loaded a separate copy of every tile for each M-tile, as the 128-token kernel does; sharing the stream halved its K/V loads. The 256-token kernel is 44% faster than our 64-token kernel at 4K tokens and 32% at 64K, from fewer, longer gathers and fewer index lookups.

![256-token blocks: one query block per CTA and one shared K/V stream](/assets/images/vsa-blackwell/fwd-blk256.svg)

*Figure 4. 256-token blocks: the CTA's two M-tiles are the two halves of one query block, so every 128-key K/V tile is loaded once and read by both.*

### 2.5 512-token blocks: two CTAs per query block, a quarter KV block per step

Blackwell's `cta_group::2` MMA lets two CTAs compute one 256-row tile together. We use it for 512-token blocks only. At 128 and 256 tokens we built 2-CTA kernels too, and they were slower than 1-CTA.

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

### 2.6 Summary

Every kernel runs two 128-lane M-tiles per CTA and ping-pongs between them: while the softmax warps work on one tile, the MMA warp runs the other tile's MMAs. TMEM fits exactly two (2 × (128 S + 128 O) = 512 columns). The block size decides what fills the two tiles and whether they can share K/V:

| Block size | Query blocks per CTA | Rows per M-tile | KV blocks per step | Top-k lists per CTA | M-tiles share K/V? |
| --- | ---: | ---: | ---: | ---: | --- |
| 64 | 2 | 64 | 4 | 2 | no |
| 128 | 2 | 128 | 1 | 2 | no |
| 256 | 1 | 128 | ½ | 1 | yes |
| 512 | ½ | 128 | ¼ | 1 | yes, across both CTAs |

KV blocks per step count one M-tile. Small blocks lose reuse because the two M-tiles hold different query blocks with different lists. From 256 tokens on, both tiles belong to one block and reuse matches dense attention, and the 512-token 2-CTA kernel doubles it.

## 3. Performance improvement

All kernel-level results use one setup:

| Component | Setup |
| --- | --- |
| GPU | NVIDIA GB200 (sm_100a), CUDA 13.0 |
| Shape | B=1, H=8, head dim 128, bf16, non-causal MHA |
| Sparsity | 25% density (top-k = number of blocks / 4), random block selection shared across heads |
| Inputs | identical saved tensors for every implementation |
| Metric | TFLOPS over selected block pairs: 4·D·H·blocks·top-k·block² |

At a fixed sequence length and density, every block size does the same number of FLOPs, so the TFLOPS of different block sizes are directly comparable. Each kernel is compared with the strongest public kernel at the same block size.

The baselines are our Triton kernel and FlashInfer at 64 tokens, FlashInfer at 128, and FA4's block-sparse kernel (our VSA-256 path) at 256, which is FA4's minimum block size. Each comparison ran in one session on identical inputs.

![TFLOPS vs sequence length](/assets/images/vsa-blackwell/fwd-perf.svg)

*Figure 6. Throughput on GB200.*

| Block | Baseline | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Triton | 3.34× | 2.74× | 2.34× | 2.25× | 2.14× | 1.92× |
| 64 | FlashInfer | 8.78× | 4.65× | 2.67× | 1.76× | 1.32× | 1.14× |
| 128 | FlashInfer | 4.21× | 2.09× | 1.26× | 1.18× | 1.13× | 1.09× |
| 256 | FA4 | 4.13× | 2.17× | 1.33× | 1.16× | 1.07× | 1.04× |

The largest gains are at short sequences, where fixed per-launch and per-tile overheads dominate the baselines.

## Acknowledgements

Thanks to the FastVideo maintainers for reviewing and merging the kernels.