"""Generate the SVG figures for the VSA-on-Blackwell blog post.

All numbers are measured TFLOPS on GB200 (B=1, H=8, D=128, bf16, 25% density),
from the FastVideo PR descriptions and the kernels' benchmark logs.
Run from anywhere: writes into assets/images/vsa-blackwell/ of this site.
"""

import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "images",
                   "vsa-blackwell")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
PANEL_BORDER = "#d6d5d0"
FONT = "Inter, 'Helvetica Neue', Arial, 'DejaVu Sans', sans-serif"

COLORS = {
    "This work": "#2a78d6",
    "FastVideo Triton": "#eb6834",
    "FlashInfer": "#1baf7a",
    "FA4": "#eda100",
}

FWD_S = ["4K", "8K", "16K", "32K", "64K", "128K"]
FWD = [
    ("64-token blocks", {
        "FastVideo Triton": [162.6, 302.6, 434.5, 484.7, 515.7, 521.9],
        "FlashInfer": [61.8, 178.2, 381.3, 618.9, 831.9, 881.6],
        "This work": [542.8, 829.2, 1016.7, 1091.3, 1102.0, 1003.0],
    }),
    ("128-token blocks", {
        "FlashInfer": [197.7, 514.3, 943.9, 1191.5, 1252.3, 1190.2],
        "This work": [832.9, 1072.7, 1184.9, 1409.9, 1414.9, 1291.9],
    }),
    ("256-token blocks", {
        "FA4": [210, 561, 972, 1333, 1434, 1387],
        "This work": [867, 1215, 1291, 1548, 1528, 1438],
    }),
]

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


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=INK_2, anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(s)}</text>')


def spread(ys, min_gap):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [ys[i] for i in order]
    for k in range(1, len(placed)):
        if placed[k] - placed[k - 1] < min_gap:
            placed[k] = placed[k - 1] + min_gap
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def small_multiples(path, panels, xlabels, ymax, ystep, series_order, title_note):
    panel_w, panel_h = 300 + 14 * max(0, len(xlabels) - 6), 250
    gap, left, top = 34, 58, 64
    label_w = 74
    plot_w = panel_w - label_w
    width = left + len(panels) * panel_w + (len(panels) - 1) * gap + 12
    height = top + panel_h + 58
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>']

    lx = left
    for name in series_order:
        c = COLORS[name]
        out.append(f'<line x1="{lx}" y1="22" x2="{lx + 22}" y2="22" stroke="{c}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{lx + 11}" cy="22" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(text(lx + 28, 26, name, 13, INK))
        lx += 28 + 8 * len(name) + 26
    out.append(text(width - 12, 26, title_note, 12, INK_3, "end"))

    for p, (ptitle, data) in enumerate(panels):
        x0 = left + p * (panel_w + gap)
        y0 = top
        out.append(text(x0, y0 - 12, ptitle, 13, INK, weight=600))
        n = len(xlabels)
        xs = [x0 + 8 + i * (plot_w - 16) / (n - 1) for i in range(n)]

        def ymap(v):
            return y0 + panel_h - v / ymax * panel_h

        v = 0
        while v <= ymax + 1e-9:
            yy = ymap(v)
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + plot_w}" y2="{yy:.1f}" '
                       f'stroke="{GRID if v else PANEL_BORDER}" stroke-width="1"/>')
            if p == 0:
                out.append(text(x0 - 8, yy + 4, f"{v:,}", 11, INK_3, "end"))
            v += ystep
        for i, lab in enumerate(xlabels):
            out.append(text(xs[i], y0 + panel_h + 18, lab, 10.5, INK_3, "middle"))

        ends = []
        for name in series_order:
            if name not in data:
                continue
            vals = data[name]
            pts = " ".join(f"{xs[i]:.1f},{ymap(val):.1f}" for i, val in enumerate(vals))
            c = COLORS[name]
            out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.25" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
            for i, val in enumerate(vals):
                out.append(f'<circle cx="{xs[i]:.1f}" cy="{ymap(val):.1f}" r="4" fill="{c}" '
                           f'stroke="{SURFACE}" stroke-width="2"><title>{esc(name)} @ S={xlabels[i]}: '
                           f'{val:,.0f} TFLOPS</title></circle>')
            ends.append((name, ymap(vals[-1]), vals[-1]))
        label_y = spread([e[1] for e in ends], 26)
        for (name, ey, ev), ly in zip(ends, label_y):
            short = {"FastVideo Triton": "Triton"}.get(name, name)
            out.append(text(xs[-1] + 10, ly - 1, short, 11.5, INK, weight=600))
            out.append(text(xs[-1] + 10, ly + 12, f"{ev:,.0f}", 11, INK_2))

    out.append(text(left - 44, top + panel_h / 2, "TFLOPS", 11, INK_3, "middle",
                    extra=f'transform="rotate(-90 {left - 44} {top + panel_h / 2})"'))
    out.append(text(left + (width - left) / 2 - 20, height - 12,
                    "Sequence length S (tokens)", 12, INK_2, "middle"))
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def box(x, y, w, h, fill="#ffffff", stroke=PANEL_BORDER, rx=6, sw=1.2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y1, x2, y2, color=INK_2, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"'
            f'{d} marker-end="url(#arrow)"/>')


ARROW_DEF = (f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             f'fill="{INK_2}"/></marker></defs>')

BLUE, BLUE_L = "#2a78d6", "#dbe8f9"
ORANGE, ORANGE_L = "#eb6834", "#fbe3d8"
AQUA_L = "#d6f1e6"


VIOLET_L = "#e7e4fa"


def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">', ARROW_DEF, f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>']


def save(path, o):
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))


def vsa_diagram(path):
    w, h = 1040, 300
    o = svg_open(w, h)
    n, cell, gx, gy = 8, 26, 70, 52
    picks = [(1, 6), (0, 3), (2, 7), (1, 6), (4, 5), (0, 2), (3, 7), (5, 6)]
    hi = 3
    o.append(text(gx + n * cell / 2, gy - 14, "KV blocks", 12, INK_2, "middle"))
    o.append(text(gx - 16, gy + n * cell / 2, "query blocks", 12, INK_2, "middle",
                  extra=f'transform="rotate(-90 {gx - 16} {gy + n * cell / 2})"'))
    for r in range(n):
        for c in range(n):
            sel = c in picks[r]
            fill = BLUE if (sel and r == hi) else ("#9ec5f4" if sel else "#ffffff")
            o.append(f'<rect x="{gx + c * cell}" y="{gy + r * cell}" width="{cell - 2}" height="{cell - 2}" '
                     f'rx="3" fill="{fill}" stroke="{PANEL_BORDER}" stroke-width="1"/>')
    o.append(f'<rect x="{gx - 3}" y="{gy + hi * cell - 3}" width="{n * cell + 4}" height="{cell + 4}" '
             f'rx="4" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    o.append(text(gx, gy + n * cell + 24, "Row 3: query block 3 attends only KV blocks 1 and 6.", 12, INK))

    tx = 360
    lines = [
        ("VSA's fine stage is block-sparse attention.", INK, 600),
        ("Tokens are grouped into blocks: 64-token (4×4×4) video cubes by default.", INK_2, 400),
        ("A coarse stage scores pooled blocks and keeps the top-k KV blocks per query block.", INK_2, 400),
        ("Each query block then attends exactly to the tokens of its own top-k list.", INK_2, 400),
        ("Neighbouring query blocks generally select different lists.", INK_2, 400),
        ("At 25% density (top-k = 2 of 8 here) the work is a quarter of dense attention.", INK_2, 400),
    ]
    for k, (s, c, wgt) in enumerate(lines):
        o.append(text(tx, 84 + k * 28, s, 13.5 if k == 0 else 13, c, weight=wgt))
    save(path, o)


def fwd_diagram(path):
    w, h = 1040, 520
    o = svg_open(w, h)
    o.append(text(30, 36, "1 · Two query blocks", 13, INK, weight=600))
    o.append(text(250, 36, "2 · 4 KV blocks per step", 13, INK, weight=600))
    o.append(text(580, 36, "3 · TMEM (128 lanes × 512 columns)", 13, INK, weight=600))

    rows = [("q block 2p", "A", BLUE_L, ("S(2p)", "cols 0–127"), ("O(2p)", "cols 256–383")),
            ("q block 2p+1", "B", ORANGE_L, ("S(2p+1)", "cols 128–255"), ("O(2p+1)", "cols 384–511"))]
    bh, bgap = 36, 6
    stack_h = 4 * bh + 3 * bgap
    sx, ox, tw = 580, 790, 150
    for r, (name, lst, fill, s_lab, o_lab) in enumerate(rows):
        y0 = 62 + r * 210
        mid = y0 + stack_h / 2
        o.append(box(30, mid - 45, 180, 90, fill))
        o.append(text(120, mid - 5, name, 13, INK, "middle", 600))
        o.append(text(120, mid + 15, f"64 rows · own top-k list {lst}", 11, INK_2, "middle"))
        o.append(arrow(210, mid, 248, mid))

        for j in range(4):
            y = y0 + j * (bh + bgap)
            o.append(box(250, y, 210, bh, AQUA_L if j < 2 else VIOLET_L, rx=4))
            o.append(text(355, y + 23, f"list {lst}, block {j + 1} · keys {64 * j}–{64 * j + 63}", 11.5, INK,
                          "middle"))
        upper = y0 + bh + bgap / 2
        lower = y0 + 3 * bh + 2.5 * bgap
        for yy in (upper, lower):
            o.append(f'<path d="M464,{yy - bh / 2 - 2} h6 v{bh + bgap + 4} h-6" fill="none" '
                     f'stroke="{INK_2}" stroke-width="1.3"/>')

        half = stack_h / 2
        for x, (lab, cols) in ((sx, s_lab), (ox, o_lab)):
            o.append(box(x, y0, tw, stack_h, fill, rx=3))
            o.append(f'<line x1="{x}" y1="{y0 + half}" x2="{x + tw}" y2="{y0 + half}" stroke="{INK_3}" '
                     f'stroke-width="1" stroke-dasharray="4 3"/>')
            o.append(text(x + tw / 2, y0 + half / 2 - 4, "lanes 0–63", 11, INK_2, "middle"))
            o.append(text(x + tw / 2, y0 + half / 2 + 13, "keys 0–127", 11.5, INK, "middle", 600))
            o.append(text(x + tw / 2, y0 + 1.5 * half - 4, "lanes 64–127", 11, INK_2, "middle"))
            o.append(text(x + tw / 2, y0 + 1.5 * half + 13, "keys 128–255", 11.5, INK, "middle", 600))
            o.append(text(x + tw / 2, y0 + stack_h + 17, f"{lab} · {cols}", 11.5, INK_2, "middle"))
        o.append(arrow(472, upper, sx - 2, y0 + half / 2))
        o.append(arrow(472, lower, sx - 2, y0 + 1.5 * half))
        o.append(arrow(sx + tw, y0 + half, ox - 2, y0 + half))
        o.append(text((sx + tw + ox) / 2, y0 + half - 8, "P·V", 11, INK_3, "middle"))

    o.append(text(30, 488, "Each S tile holds a 64 × 256 score tile in two halves (tcgen05.mma.ws, Layout E). Both query blocks use the same 128 lanes, in different columns.",
                  12, INK_2))
    o.append(text(30, 508, "Each O half is a partial output over its 128 keys; the correction warps merge the two halves of each row.",
                  12, INK_2))
    save(path, o)


def twocta_diagram(path):
    w, h = 1040, 360
    o = svg_open(w, h)
    panels = [
        ("128-token blocks", [BLUE_L, ORANGE_L, AQUA_L, VIOLET_L], "4 lists",
         ["each joint M=256 MMA spans 2 blocks", "→ must visit the union of 2 lists",
          "(2.7× the KV blocks at 25%)", "0.4–0.5× of 1-CTA"]),
        ("256-token blocks", [BLUE_L, BLUE_L, ORANGE_L, ORANGE_L], "2 lists",
         ["each joint MMA is one block, one list", "→ 2 K/V streams per cluster",
          "(same bytes per CTA as 1-CTA)", "0.91–0.97× of 1-CTA"]),
        ("512-token blocks", [BLUE_L] * 4, "1 list",
         ["the whole cluster is one block", "→ one shared K/V stream",
          "(K/V requests halved)", "best kernel at 8K and 128K"]),
    ]
    pw = 330
    for p, (title, fills, nlist, notes) in enumerate(panels):
        x0 = 30 + p * (pw + 10)
        o.append(text(x0, 34, title, 13, INK, weight=600))
        bx, by, seg = x0 + 6, 52, 52
        for s, fill in enumerate(fills):
            o.append(f'<rect x="{bx}" y="{by + s * seg}" width="96" height="{seg - 2}" fill="{fill}" '
                     f'stroke="{PANEL_BORDER}" stroke-width="1"/>')
            o.append(text(bx + 48, by + s * seg + 30, f"CTA {s % 2} · 128 rows", 10.5, INK_2, "middle"))
        for m in range(2):
            y1, y2 = by + m * 2 * seg, by + (m + 1) * 2 * seg - 2
            o.append(f'<path d="M{bx + 104},{y1} h8 v{y2 - y1} h-8" fill="none" stroke="{INK_2}" stroke-width="1.4"/>')
            o.append(text(bx + 118, (y1 + y2) / 2 + 4, f"joint MMA {m}", 11, INK_2))
        o.append(text(bx + 118, by + 4 * seg + 2, f"512 rows · {nlist}", 11, INK_3))
        for k, s in enumerate(notes):
            last = k == len(notes) - 1
            o.append(text(x0 + 6, 288 + k * 18, s, 12 if last else 11.5, INK if last else INK_2,
                          weight=600 if last else 400))
    save(path, o)


def bwd_diagram(path):
    w, h = 1040, 340
    o = svg_open(w, h)
    o.append(box(30, 40, 210, 150, "#ffffff"))
    o.append(text(135, 64, "Preprocess", 13, INK, "middle", 600))
    for k, s in enumerate(["Δ = rowsum(dO ∘ O)", "zero the fp32 dQ accumulator",
                           "zero dK, dV of unselected", "KV blocks; work order"]):
        o.append(text(135, 92 + k * 20, s, 11.5, INK_2, "middle"))

    o.append(box(300, 40, 440, 236, BLUE_L, BLUE))
    o.append(text(520, 64, "Main: one CTA per KV block (KV-stationary)", 13, INK, "middle", 600))
    o.append(text(520, 86, "K, V loaded once; loop over the query blocks that selected it", 11.5, INK_2,
                  "middle"))
    steps = ["Sᵀ = K Qᵀ  →  Pᵀ = exp2(scale · Sᵀ − LSE)",
             "dPᵀ = V dOᵀ  →  dSᵀ = Pᵀ ∘ (dPᵀ − Δ)",
             "dV += Pᵀ dO;  dK += dSᵀ Q   (accumulated in TMEM)",
             "dQ += dS K  →  fp32 reduce-add to global memory"]
    for k, s in enumerate(steps):
        y = 104 + k * 40
        o.append(box(320, y, 400, 32, "#ffffff", PANEL_BORDER, 4))
        o.append(text(332, y + 21, s, 12, INK, extra='xml:space="preserve"'))

    o.append(box(800, 40, 210, 150, "#ffffff"))
    o.append(text(905, 64, "Postprocess", 13, INK, "middle", 600))
    for k, s in enumerate(["dQ = scale · accumulator", "fp32 → bf16"]):
        o.append(text(905, 92 + k * 20, s, 11.5, INK_2, "middle"))

    o.append(arrow(240, 115, 298, 115))
    o.append(arrow(740, 115, 798, 115))
    o.append(text(269, 106, "PDL", 10.5, INK_3, "middle"))
    o.append(text(769, 106, "PDL", 10.5, INK_3, "middle"))

    o.append(box(30, 216, 210, 104, ORANGE_L, ORANGE))
    o.append(text(135, 238, "64-token blocks, S ≤ 128K", 12.5, INK, "middle", 600))
    for k, s in enumerate(["pass 1: dK, dV; store dS tiles", "pass 2: dQ from stored tiles",
                           "no fp32 read-modify-write"]):
        o.append(text(135, 262 + k * 18, s, 11, INK_2, "middle"))
    o.append(box(800, 216, 210, 104, "#ffffff"))
    o.append(text(905, 238, "256-token blocks: 2-CTA", 12.5, INK, "middle", 600))
    for k, s in enumerate(["a CTA pair covers the 256-row", "KV block with M=256 MMAs;",
                           "each loads half of Q, dO"]):
        o.append(text(905, 262 + k * 18, s, 11, INK_2, "middle"))
    save(path, o)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    small_multiples(os.path.join(OUT, "fwd-perf.svg"), FWD, FWD_S, 1600, 400,
                    ["This work", "FastVideo Triton", "FlashInfer", "FA4"],
                    "Forward · GB200 · H=8, D=128, 25% density")
    small_multiples(os.path.join(OUT, "bwd-perf.svg"), BWD, BWD_S, 1600, 400,
                    ["This work", "FastVideo Triton", "FA4"],
                    "Backward · GB200 · H=8, D=128, 25% density")
    vsa_diagram(os.path.join(OUT, "vsa-fine-stage.svg"))
    fwd_diagram(os.path.join(OUT, "fwd-dualpack.svg"))
    twocta_diagram(os.path.join(OUT, "fwd-2cta.svg"))
    bwd_diagram(os.path.join(OUT, "bwd-pipeline.svg"))
    print("wrote", sorted(os.listdir(OUT)))
