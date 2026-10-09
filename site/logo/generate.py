#!/bin/env python
"""Skoro logo generator: stitched eight-pointed star with posterior-tail samples,
plus the word "skoro" drawn as a cadlag path in R^2. Standard library only.

Writes skoro-mark.svg, skoro-lockup.svg and the site's favicon (../static/favicon.svg). All geometry is in stitch units (1 unit = 1 cell).
"""

import math
import random

# ---- settings ----
COLOR, INK, PAPER = "#7a5cd6", "#1c1c1c", "#ffffff"
STAR_R = 11  # half-width of the star's grid
TAIL = 4  # extra cells of room for tail samples
SEED, N_SAMPLES = 3, 14
GAP, SCALE = 1.0, 1.15  # text gap from star edge (stitches), text scale
SHOW_JUMPS = True  # dashed lines along each pen-lift

M = STAR_R + TAIL
H = 2 * M + 1


# ---- star ----
def radius(dx, dy):
    """Smallest r such that cell (dx, dy) lies in the eight-pointed star of size r."""
    return min(max(dx, dy), math.ceil((dx + dy) / 1.5))


def star_cells():
    cells = []
    for i in range(-STAR_R, STAR_R + 1):
        for j in range(-STAR_R, STAR_R + 1):
            dx, dy = abs(i), abs(j)
            r = radius(dx, dy)
            if 5 < r <= 7:
                cells.append((i, j, COLOR, 1.0))
            elif 2 < r <= 3:
                cells.append((i, j, INK, 1.0))
            elif dx + dy <= 1:
                cells.append((i, j, COLOR, 1.0))
    return cells


def tail_samples():
    """Draws in the star's tails: weight decays with distance past the colored band, following its contours."""
    rng = random.Random(SEED)
    cand = []
    for i in range(-M, M + 1):
        for j in range(-M, M + 1):
            e = radius(abs(i), abs(j)) - 7
            if 2 <= e <= 4:
                cand.append(((i, j), e, math.exp(-(e - 2) / 0.9)))
    opacity = {2: 0.7, 3: 0.45, 4: 0.28}
    out = []
    while len(out) < N_SAMPLES:
        u = rng.random() * sum(w for _, _, w in cand)
        for k, ((i, j), e, w) in enumerate(cand):
            u -= w
            if u <= 0:
                out.append((i, j, COLOR, opacity[e]))
                cand.pop(k)
                break
    return out


def stitch(x, y, col, op=1.0, pad=0.17):
    a, b, c, d = x + pad, x + 1 - pad, y + pad, y + 1 - pad
    o = "" if op >= 0.999 else f' stroke-opacity="{op:.2f}"'
    return f'<path d="M{a:.2f},{c:.2f}L{b:.2f},{d:.2f}M{b:.2f},{c:.2f}L{a:.2f},{d:.2f}" stroke="{col}"{o}/>'


def mark_svg(cells):
    body = "\n".join(stitch(i + M, j + M, col, op) for i, j, col, op in cells)
    return f'<g fill="none" stroke-linecap="round" stroke-width="0.24">\n{body}\n</g>'


# ---- the word as a cadlag path: continuous strokes, pen-lifts are jumps ----
# glyph units: x-height 6, ascender 9, baseline y=0, up is negative.
# each stroke: (path d, start point, end point)
O = (
    "M3.6,-5.2 C2.6,-6.6 0,-6 0,-3 C0,0.9 4.4,0.9 4.4,-3 C4.4,-4.6 3.4,-6.4 1.2,-5.8",
    (3.6, -5.2),
    (1.2, -5.8),
)
GLYPHS = [
    (
        4.0,
        [
            (
                "M4,-5.1 C3.4,-6.1 0.3,-6.2 0.3,-4.5 C0.3,-3 4,-3.1 4,-1.5 C4,0.3 0.8,0.2 0,-0.9",
                (4, -5.1),
                (0, -0.9),
            )
        ],
    ),  # s
    (
        4.0,
        [
            ("M0,-9 L0,0", (0, -9), (0, 0)),  # k stem
            ("M3.8,-6 L0.4,-2.8 L4,0", (3.8, -6), (4, 0)),
        ],
    ),  # k arm
    (4.4, [O]),  # o
    (3.4, [("M0,0 L0,-3.6 C0,-5.5 1.8,-6.2 3.4,-5.5", (0, 0), (3.4, -5.5))]),  # r
    (4.4, [O]),  # o
]


def word_svg(x0, base, sc, spacing=1.5, lw=0.62):
    strokes, x = [], x0
    for w, ss in GLYPHS:
        for d, a, b in ss:
            strokes.append(
                (
                    d,
                    x,
                    (x + a[0] * sc, base + a[1] * sc),
                    (x + b[0] * sc, base + b[1] * sc),
                )
            )
        x += (w + spacing) * sc
    out = []
    if SHOW_JUMPS:
        for (_, _, _, a), (_, _, b, _) in zip(strokes, strokes[1:]):
            out.append(
                f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{COLOR}" '
                f'stroke-opacity="0.45" stroke-width="0.16" stroke-dasharray="0.35 0.45"/>'
            )
    for d, ox, _, _ in strokes:
        out.append(
            f'<path d="{d}" transform="translate({ox:.2f},{base:.2f}) scale({sc})" fill="none" stroke="{INK}" '
            f'stroke-width="{lw / sc:.3f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for k, (_, _, a, b) in enumerate(strokes):
        out.append(
            f'<circle cx="{a[0]:.2f}" cy="{a[1]:.2f}" r="0.5" fill="{COLOR}"/>'
        )  # value at jump time
        if k < len(strokes) - 1:
            out.append(
                f'<circle cx="{b[0]:.2f}" cy="{b[1]:.2f}" r="0.42" fill="{PAPER}" '  # left limit
                f'stroke="{COLOR}" stroke-width="0.2"/>'
            )
    width = x - spacing * sc - x0
    return "\n".join(out), width


def favicon_svg(cells, width=0.42, dark_ink="#e6e6ea"):
    """The star alone, cropped tight, with heavier stitches so it survives 16px; the ink band lightens
    on a dark tab strip."""
    body = "".join(
        stitch(i + STAR_R, j + STAR_R, col, op, pad=0.12)[:-2] + f' stroke-width="{width}"/>'
        for i, j, col, op in cells
    )
    n = 2 * STAR_R + 1
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}">'
        f'<style>@media (prefers-color-scheme: dark) {{ [stroke="{INK}"] {{ stroke: {dark_ink}; }} }}</style>'
        f'<g fill="none" stroke-linecap="round">{body}</g></svg>\n'
    )


def svg(w, h, inner):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h}" width="{w * 10:.0f}" height="{h * 10}">\n'
        f"<title>Skoro</title>\n{inner}\n</svg>\n"
    )


def main():
    stars, samples = star_cells(), tail_samples()

    with open("skoro-mark.svg", "w") as f:
        f.write(svg(H, H, mark_svg(samples + stars)))

    x0 = M + 0.5 + STAR_R + 0.5 + GAP  # star's right edge + gap
    base = M + 0.5 + 3.0 * SCALE  # x-height centred on the star's centre row
    word, wd = word_svg(x0, base, SCALE)
    kept = [
        s for s in samples if s[0] + M + 1 <= x0 - 0.6
    ]  # drop tail stitches that would sit under the text
    with open("skoro-lockup.svg", "w") as f:
        f.write(svg(x0 + wd + 1.5, H, mark_svg(kept + stars) + "\n" + word))

    with open("../static/favicon.svg", "w") as f:
        f.write(favicon_svg(stars))


if __name__ == "__main__":
    main()
