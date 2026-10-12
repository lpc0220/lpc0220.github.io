"""Generate the SVG figures for the VSA backward blog post.

Shares the drawing helpers and palette of ../vsa-blackwell/make_figures.py.
All numbers are measured TFLOPS on GB200 (B=1, H=8, D=128, bf16, 25% density).
Run from anywhere: writes into assets/images/vsa-blackwell-backward/ of this site.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vsa-blackwell"))

from make_figures import (AQUA_L, BLUE, BLUE_L, INK, INK_2, INK_3, ORANGE, ORANGE_L,  # noqa: E402
                          PANEL_BORDER, VIOLET_L, arrow, box, save, small_multiples, svg_open,
                          text)

OUT = os.path.join(HERE, "..", "..", "assets", "images", "vsa-blackwell-backward")

BWD_S = ["4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K"]
BWD = [
    ("64-token blocks", {
        "FastVideo Triton": [198, 313, 389, 433, 448, 452, 452, 437],
        "This work": [465, 688, 860, 918, 857, 816, 767, 755],
    }),
    ("128-token blocks", {
        "FA4": [266, 608, 999, 1260, 1155, 1025, 892, 828],
        "This work": [598, 893, 1133, 1332, 1183, 1055, 923, 850],
    }),
    ("256-token blocks", {
        "FA4": [172, 504, 954, 1332, 1278, 1219, 1104, 1057],
        "This work": [639, 952, 1222, 1422, 1272, 1180, 1084, 1035],
    }),
]

GRAY = "#eeeeeb"


def lists_diagram(path):
    w, h = 1040, 300
    o = svg_open(w, h)
    n, cell, gx, gy = 8, 26, 70, 52
    picks = [(1, 3), (0, 3), (3, 6), (1, 6), (0, 6), (2, 7), (5, 6), (5, 7)]
    row, col = 3, 6
    o.append(text(gx + n * cell / 2, gy - 14, "KV blocks", 12, INK_2, "middle"))
    o.append(text(gx - 16, gy + n * cell / 2, "query blocks", 12, INK_2, "middle",
                  extra=f'transform="rotate(-90 {gx - 16} {gy + n * cell / 2})"'))
    for r in range(n):
        for c in range(n):
            sel = c in picks[r]
            fill = "#ffffff"
            if sel:
                fill = BLUE if r == row else (ORANGE if c == col else "#9ec5f4")
            o.append(f'<rect x="{gx + c * cell}" y="{gy + r * cell}" width="{cell - 2}" height="{cell - 2}" '
                     f'rx="3" fill="{fill}" stroke="{PANEL_BORDER}" stroke-width="1"/>')
    o.append(f'<rect x="{gx - 3}" y="{gy + row * cell - 3}" width="{n * cell + 4}" height="{cell + 4}" '
             f'rx="4" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    o.append(f'<rect x="{gx + col * cell - 3}" y="{gy - 3}" width="{cell + 4}" height="{n * cell + 4}" '
             f'rx="4" fill="none" stroke="{ORANGE}" stroke-width="2"/>')
    counts = [sum(c in p for p in picks) for c in range(n)]
    for c, k in enumerate(counts):
        o.append(text(gx + c * cell + cell / 2 - 1, gy + n * cell + 18, str(k), 12,
                      INK if c == col else INK_2, "middle", 600 if c == col else 400))
    o.append(text(gx - 8, gy + n * cell + 18, "count", 11, INK_3, "end"))

    tx = 360
    lines = [
        ("The backward reads the same mask in both directions.", INK, 600),
        ("Row 3 (blue): query block 3 attends KV blocks 1 and 6, its top-k list.", INK_2, 400),
        ("Column 6 (orange): KV block 6 was selected by query blocks 2, 3, 4 and 6.", INK_2, 400),
        ("dQ of a query block sums over its row; dK and dV of a KV block sum over its column.", INK_2, 400),
        ("Every row has exactly top-k entries, but column counts vary (here 0 to 4).", INK_2, 400),
        ("KV block 4 was selected by no query block: its dK and dV are zero.", INK_2, 400),
    ]
    for k, (s, c, wgt) in enumerate(lines):
        o.append(text(tx, 84 + k * 28, s, 13.5 if k == 0 else 13, c, weight=wgt))
    save(path, o)


def blk128_diagram(path):
    w, h = 1040, 590
    o = svg_open(w, h)
    o.append(text(30, 36, "1 · Shared memory", 13, INK, weight=600))
    o.append(text(250, 36, "2 · Five GEMMs per step", 13, INK, weight=600))
    o.append(text(640, 36, "3 · TMEM (128 lanes × 512 columns)", 13, INK, weight=600))

    smem = [("K · 128 KV rows", "loaded once", AQUA_L),
            ("V · 128 KV rows", "loaded once", AQUA_L),
            ("Q · 128 query rows", "2 stages, one block per step", BLUE_L),
            ("dO · 128 query rows", "1 stage", BLUE_L),
            ("dS (bf16)", "written by the compute warps", ORANGE_L)]
    gemms = [("1 · Sᵀ = K · Qᵀ", "→ cols 0–127"),
             ("2 · dPᵀ = V · dOᵀ", "→ cols 256–383"),
             ("3 · dV += Pᵀ · dO", "Pᵀ read from TMEM → cols 128–255"),
             ("4 · dQ = dS · K", "dS read from SMEM → cols 256–383"),
             ("5 · dK += dSᵀ · Q", "dSᵀ read from TMEM → cols 384–511")]
    for i, ((name, sub, fill), (g, gsub)) in enumerate(zip(smem, gemms)):
        y = 62 + i * 62
        o.append(box(30, y, 190, 52, fill, rx=4))
        o.append(text(125, y + 22, name, 12, INK, "middle", 600))
        o.append(text(125, y + 40, sub, 11, INK_2, "middle"))
        o.append(box(250, y, 350, 52, "#ffffff", rx=4))
        o.append(text(266, y + 22, g, 12.5, INK, weight=600))
        o.append(text(266, y + 40, gsub, 11, INK_2))

    tmem = [("cols 0–127", "Sᵀ, then Pᵀ (bf16) written over it", BLUE_L),
            ("cols 128–255", "dV accumulator, kept for the whole list", VIOLET_L),
            ("cols 256–383", "dPᵀ, then dSᵀ (bf16), then the step's dQ tile", ORANGE_L),
            ("cols 384–511", "dK accumulator, kept for the whole list", VIOLET_L)]
    for i, (cols, body, fill) in enumerate(tmem):
        y = 62 + i * 78
        o.append(box(640, y, 370, 70, fill, rx=3))
        o.append(text(656, y + 30, cols, 12.5, INK, weight=600))
        o.append(text(656, y + 50, body, 11.5, INK_2))

    y0 = 392
    o.append(box(30, y0, 400, 92, "#ffffff"))
    o.append(text(46, y0 + 24, "8 compute warps", 12.5, INK, weight=600))
    for k, s in enumerate(["Pᵀ = exp2(Sᵀ · scale − LSE)", "dSᵀ = Pᵀ ∘ (dPᵀ − Δ)",
                           "write dSᵀ to TMEM (GEMM 5) and dS to SMEM (GEMM 4)"]):
        o.append(text(46, y0 + 46 + k * 18, s, 11.5, INK_2))
    o.append(box(460, y0, 300, 92, "#ffffff"))
    o.append(text(476, y0 + 24, "4 dQ warps", 12.5, INK, weight=600))
    for k, s in enumerate(["read the dQ tile out of TMEM", "4 chunks × 16 KB, 2 SMEM buffers",
                           "fp32 bulk reduce-add per chunk"]):
        o.append(text(476, y0 + 46 + k * 18, s, 11.5, INK_2))
    o.append(arrow(760, y0 + 46, 798, y0 + 46))
    o.append(box(800, y0, 210, 92, ORANGE_L, rx=4))
    o.append(text(905, y0 + 34, "Global memory", 12.5, INK, "middle", 600))
    o.append(text(905, y0 + 54, "fp32 dQ accumulator,", 11.5, INK_2, "middle"))
    o.append(text(905, y0 + 72, "summed over KV blocks", 11.5, INK_2, "middle"))

    o.append(text(30, 528, "Issue order for step j: Sᵀ(j+1) → dK(j) → dQ(j) → dPᵀ(j+1) → dV(j+1). The compute warps start step j+1 while the tensor core finishes step j.",
                  12, INK_2))
    o.append(text(30, 548, "dPᵀ(j+1) waits until the dQ warps have read dQ(j) out of cols 256–383. dK(j) reads dSᵀ there before dQ(j) overwrites it: MMAs run in issue order.",
                  12, INK_2))
    o.append(text(30, 568, "K and V stay in shared memory and dK and dV in TMEM for the whole list; only dQ leaves the SM every step.",
                  12, INK_2))
    save(path, o)


def qtile(x, y, label, fill, w=40, h=28, dashed=False):
    d = ' stroke-dasharray="4 3"' if dashed else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{INK_3 if dashed else PANEL_BORDER}" '
            f'stroke-width="1.2"{d}/>' + text(x + w / 2, y + 18, label, 12, INK, "middle"))


def blk64_diagram(path):
    w, h = 1040, 470
    o = svg_open(w, h)
    o.append(text(30, 36, "1 · List walked in quads", 13, INK, weight=600))
    o.append(text(330, 36, "2 · Ring: 4 slots × 32 KB", 13, INK, weight=600))
    o.append(text(650, 36, "3 · TMEM: Sᵀ of one quad (64 KV × 256 query)", 13, INK, weight=600))

    o.append(text(30, 70, "KV block j · 64 rows · list:", 11.5, INK_2))
    for k, q in enumerate([8, 11, 20, 3, 7, 15]):
        o.append(qtile(30 + k * 44, 80, str(q), BLUE_L))
    for qd, (ids, pads) in enumerate([([8, 11, 20, 3], 0), ([7, 15, 15, 15], 2)]):
        y = 140 + qd * 74
        o.append(text(30, y - 6, f"quad {qd}", 11.5, INK, weight=600))
        for k, q in enumerate(ids):
            pad = k >= 4 - pads
            o.append(qtile(30 + k * 44, y, str(q), "#ffffff" if pad else BLUE_L, dashed=pad))
            o.append(text(30 + k * 44 + 20, y + 42, f"b{k}", 10.5, INK_3, "middle"))
    o.append(text(30, 300, "dashed = pad (repeats the last block);", 11, INK_3))
    o.append(text(30, 316, "the compute warps set P = dS = 0 there", 11, INK_3))

    slots = [("s0 · dOᵀ, b0 b1", "freed by dV"), ("s1 · dOᵀ, b2 b3", "freed by dV"),
             ("s2 · Qᵀ, b0 b1", "freed by dK"), ("s3 · Qᵀ, b2 b3", "freed by dK")]
    for k, (lab, sub) in enumerate(slots):
        y = 62 + k * 58
        o.append(box(330, y, 230, 48, BLUE_L, rx=4))
        o.append(text(346, y + 21, lab, 12, INK, weight=600))
        o.append(text(346, y + 38, sub, 11, INK_2))
    o.append(text(330, 314, "K and V: loaded once per CTA.", 11, INK_3))

    sx, sy, sw, sh = 650, 62, 360, 160
    o.append(box(sx, sy, sw, sh, "#ffffff", rx=3))
    o.append(f'<line x1="{sx}" y1="{sy + sh / 2}" x2="{sx + sw}" y2="{sy + sh / 2}" stroke="{INK_3}" '
             f'stroke-width="1" stroke-dasharray="4 3"/>')
    o.append(f'<line x1="{sx + sw / 2}" y1="{sy}" x2="{sx + sw / 2}" y2="{sy + sh}" stroke="{INK_3}" '
             f'stroke-width="1" stroke-dasharray="4 3"/>')
    cells = [(0, 0, "b0", AQUA_L), (1, 0, "b2", VIOLET_L), (0, 1, "b1", AQUA_L), (1, 1, "b3", VIOLET_L)]
    for cx, cy, lab, fill in cells:
        x = sx + cx * sw / 2
        y = sy + cy * sh / 2
        o.append(f'<rect x="{x + 4}" y="{y + 4}" width="{sw / 2 - 8}" height="{sh / 2 - 8}" rx="3" fill="{fill}"/>')
        o.append(text(x + sw / 4, y + sh / 4 - 2, f"query block {lab}", 12, INK, "middle", 600))
        o.append(text(x + sw / 4, y + sh / 4 + 16, "64 KV × 64 query", 11, INK_2, "middle"))
    o.append(text(sx - 8, sy + sh / 4 + 4, "lanes 0–63", 11, INK_2, "end"))
    o.append(text(sx - 8, sy + 3 * sh / 4 + 4, "64–127", 11, INK_2, "end"))
    o.append(text(sx + sw / 4, sy + sh + 18, "cols 0–63", 11, INK_2, "middle"))
    o.append(text(sx + 3 * sw / 4, sy + sh + 18, "cols 64–127", 11, INK_2, "middle"))
    o.append(text(sx, sy + sh + 46, "dV and dK use the same lane split: lanes 0–63", 11.5, INK_2))
    o.append(text(sx, sy + sh + 64, "sum over b0, b2, lanes 64–127 over b1, b3;", 11.5, INK_2))
    o.append(text(sx, sy + sh + 82, "the two partials are added at the end.", 11.5, INK_2))

    o.append(text(30, 372, "Each 128-lane tcgen05.mma.ws (Layout E) computes two independent 64-row products, so a 64-row KV block still uses the full tensor core.",
                  12, INK_2))
    o.append(text(30, 392, "Sᵀ and dPᵀ are issued as two N=128 MMAs, one per Qᵀ or dOᵀ slot; dV and dK accumulate over slot s0 then s1 (s2 then s3).",
                  12, INK_2))
    o.append(text(30, 412, "dQ = dS · K runs as two plain 128-row MMAs per quad (b0 and b2, then b1 and b3); each dQ tile goes to the fp32 accumulator.",
                  12, INK_2))
    o.append(text(30, 432, "The loader fills the next quad's dOᵀ slots first: dV frees them before dK frees the Qᵀ slots.",
                  12, INK_2))
    save(path, o)


def flow_box(o, x, y, w, h, title, lines, fill="#ffffff"):
    o.append(box(x, y, w, h, fill, rx=4))
    o.append(text(x + w / 2, y + 22, title, 12, INK, "middle", 600))
    for k, s in enumerate(lines):
        o.append(text(x + w / 2, y + 40 + k * 16, s, 11, INK_2, "middle"))


def twopass_diagram(path):
    w, h = 1040, 400
    o = svg_open(w, h)
    o.append(text(30, 36, "One pass (64-token blocks, from 256K tokens)", 13, INK, weight=600))
    y = 52
    flow_box(o, 30, y, 210, 76, "Main · CTA per KV block", ["dK, dV in TMEM", "dQ partial per pair"], AQUA_L)
    o.append(arrow(240, y + 38, 318, y + 38))
    o.append(text(279, y + 28, "32 KB fp32", 10.5, INK_3, "middle"))
    flow_box(o, 320, y, 230, 76, "fp32 dQ accumulator", ["reduce-add per pair:", "read-modify-write in L2"], ORANGE_L)
    o.append(arrow(550, y + 38, 618, y + 38))
    flow_box(o, 620, y, 180, 76, "Postprocess", ["× scale, bf16 dQ"])

    o.append(text(30, 172, "Two passes (64-token blocks, up to 128K tokens)", 13, INK, weight=600))
    y = 188
    flow_box(o, 30, y, 210, 76, "Pass 1 · CTA per KV block", ["dK, dV in TMEM", "store each dSᵀ tile"], AQUA_L)
    o.append(arrow(240, y + 38, 318, y + 38))
    o.append(text(279, y + 28, "8 KB bf16", 10.5, INK_3, "middle"))
    flow_box(o, 320, y, 230, 76, "dS buffer (global)", ["one 64 × 64 tile per pair,", "at (query block, list position)"], ORANGE_L)
    o.append(arrow(550, y + 38, 618, y + 38))
    o.append(text(584, y + 28, "8 KB dS", 10.5, INK_3, "middle"))
    o.append(text(584, y + 56, "16 KB K", 10.5, INK_3, "middle"))
    flow_box(o, 620, y, 210, 76, "Pass 2 · CTA per query block", ["walks its own top-k list", "dQ += dS · K in TMEM"], BLUE_L)
    o.append(arrow(830, y + 38, 878, y + 38))
    flow_box(o, 880, y, 130, 76, "bf16 dQ", ["stored once"])

    o.append(text(30, 312, "Pass 2 needs no recomputation (dS is stored) and no read-modify-write: each query block sums its own dQ in TMEM.",
                  12, INK_2))
    o.append(text(30, 332, "The cost is the dS buffer: B · H · blocks · top-k · 8 KB, 68.7 GB at 128K tokens with 8 heads. At long sequences it is filled",
                  12, INK_2))
    o.append(text(30, 352, "and consumed one head at a time; past 128K the per-head data no longer fits in L2 and one pass is faster.",
                  12, INK_2))
    save(path, o)


def blk256_diagram(path):
    w, h = 1040, 430
    o = svg_open(w, h)
    pw = 480
    panels = [
        ("1-CTA: two independent CTAs per KV block", False),
        ("2-CTA pair: one KV block, M = 256 MMAs", True),
    ]
    for p, (title, pair) in enumerate(panels):
        x0 = 30 + p * (pw + 20)
        o.append(text(x0, 36, title, 13, INK, weight=600))
        o.append(box(x0, 52, pw, 300, "#ffffff"))
        if not pair:
            for c in range(2):
                bx = x0 + 16 + c * 228
                o.append(box(bx, 68, 212, 60, AQUA_L, rx=4))
                o.append(text(bx + 106, 92, f"CTA {c} · KV rows {128 * c}–{128 * c + 127}", 12, INK, "middle", 600))
                o.append(text(bx + 106, 112, "M = 128 MMAs", 11, INK_2, "middle"))
            notes = ["Both CTAs walk the same list of query tiles (128 rows each).",
                     "Each loads every Q and dO tile in full.",
                     "Each CTA's dQ partial sums over its own 128 keys:",
                     "two 64 KB fp32 reduce-adds per query tile."]
        else:
            o.append(box(x0 + 16, 68, pw - 32, 60, AQUA_L, rx=4))
            o.append(f'<line x1="{x0 + pw / 2}" y1="72" x2="{x0 + pw / 2}" y2="100" stroke="{INK_3}" '
                     f'stroke-width="1" stroke-dasharray="4 3"/>')
            o.append(text(x0 + 16 + (pw - 32) / 4, 92, "CTA 0 · KV rows 0–127", 12, INK, "middle", 600))
            o.append(text(x0 + 16 + 3 * (pw - 32) / 4, 92, "CTA 1 · KV rows 128–255", 12, INK, "middle", 600))
            o.append(text(x0 + pw / 2, 116, "joint M = 256 MMAs over the whole KV block", 11, INK_2, "middle"))
            notes = ["Each CTA loads half of each Q and dO tile (in both layouts).",
                     "The CTAs exchange their dS halves through distributed SMEM.",
                     "dQ = dS · K contracts all 256 keys in one cluster MMA;",
                     "each CTA pushes its own 64 query rows: one reduce-add",
                     "per query tile, in two disjoint 32 KB halves."]
        for k, s in enumerate(notes):
            o.append(text(x0 + 18, 160 + k * 22, s, 12, INK_2))
        o.append(arrow(x0 + pw / 2, 280, x0 + pw / 2, 300))
        o.append(box(x0 + 110, 302, pw - 220, 40, ORANGE_L, rx=4))
        o.append(text(x0 + pw / 2, 327, "fp32 dQ accumulator", 12, INK, "middle", 600))

    o.append(text(30, 384, "The pair halves the dQ reduce-add traffic per FLOP; Q and dO bytes per CTA stay the same.",
                  12, INK_2))
    o.append(text(30, 404, "Measured: 5–13% faster than the 1-CTA form from 4K to 512K tokens, the largest gain at 512K.",
                  12, INK_2))
    save(path, o)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lists_diagram(os.path.join(OUT, "bwd-lists.svg"))
    blk128_diagram(os.path.join(OUT, "bwd-blk128.svg"))
    blk64_diagram(os.path.join(OUT, "bwd-blk64.svg"))
    twopass_diagram(os.path.join(OUT, "bwd-twopass.svg"))
    blk256_diagram(os.path.join(OUT, "bwd-blk256.svg"))
    small_multiples(os.path.join(OUT, "bwd-perf.svg"), BWD, BWD_S, 1600, 400,
                    ["This work", "FastVideo Triton", "FA4"],
                    "GB200 · H=8, D=128, 25% density")
    print("wrote", sorted(os.listdir(OUT)))
