#!/usr/bin/env python3
"""
Soul Sky preview renderer — domain-warp fractal nebula + physical stars.
Targets the same visual quality as the React Native / Skia implementation.

Usage:
    python3 client/preview_render.py [--out ~/Desktop/soul_sky_preview.png]
"""

import argparse, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

W = H = 1080

# ─── Harry Potter soul data ───────────────────────────────────────────────────

NEBULAS = [
    # cx    cy    rx    ry    color (R,G,B 0-1)          intensity  seed
    dict(cx=0.42, cy=0.45, rx=0.38, ry=0.34, color=(0.22, 0.32, 0.90), intensity=1.0, seed=12),  # loneliness — deep blue
    dict(cx=0.60, cy=0.52, rx=0.30, ry=0.26, color=(0.14, 0.08, 0.55), intensity=0.85, seed=37),  # grief — dark indigo
    dict(cx=0.30, cy=0.60, rx=0.26, ry=0.22, color=(0.48, 0.28, 0.92), intensity=0.80, seed=71),  # purpose — violet
    dict(cx=0.68, cy=0.35, rx=0.22, ry=0.20, color=(0.85, 0.22, 0.12), intensity=0.65, seed=55),  # danger — red
    dict(cx=0.55, cy=0.75, rx=0.18, ry=0.16, color=(0.15, 0.55, 0.80), intensity=0.55, seed=90),  # longing — sky blue
]

SURFACE_STARS = [
    dict(x=0.500, y=0.500, mag=9.5, color=(1.00, 0.95, 0.85), label="当下",    soul=True),
    dict(x=0.660, y=0.500, mag=7.2, color=(1.00, 0.69, 0.78), label="关系"),
    dict(x=0.580, y=0.361, mag=6.8, color=(1.00, 0.82, 0.50), label="事业"),
    dict(x=0.420, y=0.361, mag=5.5, color=(0.50, 1.00, 0.69), label="健康"),
    dict(x=0.340, y=0.500, mag=7.5, color=(0.82, 0.75, 1.00), label="使命"),
    dict(x=0.420, y=0.639, mag=5.2, color=(0.50, 0.91, 1.00), label="身份"),
    dict(x=0.820, y=0.180, mag=7.8, color=(1.00, 0.91, 0.75), label="邓布利多"),
]

CONSTELLATION = [
    (0.500, 0.500),
    (0.660, 0.500),
    (0.580, 0.361),
    (0.340, 0.500),
    (0.420, 0.639),
]
CONST_EDGES = [(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)]

PARAMS = dict(
    bgDensity=1.4, nebulaGamma=0.72, colorTemp=-0.28,
    vigStrength=0.42, soulGlow=1.5,
)

# ─── Noise primitives (vectorised) ───────────────────────────────────────────

def _hash2(px, py):
    hx = np.sin(px * 127.1 + py * 311.7) * 43758.5453
    hy = np.sin(px * 269.5 + py * 183.3) * 43758.5453
    return hx - np.floor(hx), hy - np.floor(hy)

def vnoise(xs, ys):
    ix, iy   = np.floor(xs), np.floor(ys)
    fx, fy   = xs - ix, ys - iy
    ux = fx*fx*(3 - 2*fx); uy = fy*fy*(3 - 2*fy)
    def g(di, dj):
        hx, hy = _hash2(ix + di, iy + dj)
        gx = hx*2 - 1; gy = hy*2 - 1
        return gx*(fx - di) + gy*(fy - dj)
    return (g(0,0)*(1-ux) + g(1,0)*ux)*(1-uy) + (g(0,1)*(1-ux) + g(1,1)*ux)*uy

def fbm5(xs, ys):
    v, a, f = np.zeros_like(xs, dtype=np.float32), 0.55, 1.8
    x_, y_  = xs.copy(), ys.copy()
    for _ in range(5):
        v += a * (vnoise(x_*f, y_*f)*0.5 + 0.5)
        f *= 2.1; a *= 0.48
    return v

def turb5(xs, ys):
    t, a, f = np.zeros_like(xs, dtype=np.float32), 0.52, 1.6
    for _ in range(5):
        t += a * np.abs(vnoise(xs*f, ys*f))
        f *= 2.05; a *= 0.50
    return t

# ─── Layer builders ───────────────────────────────────────────────────────────

def build_starfield(density=1.2, color_temp=0.0, seed=42) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000) % (2**31))
    img = np.zeros((H, W, 3), np.float32)

    # Fine background haze — subtle overall luminosity
    gx, gy = np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
    band = np.exp(-((gy - 0.42)*3.5)**2) * 0.012 * density
    img[:,:,0] += band * 0.8; img[:,:,1] += band * 0.85; img[:,:,2] += band

    # Individual stars
    n = int(5000 * density)
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    bright = (rng.exponential(0.35, n)**1.2).clip(0, 1)
    temps  = rng.uniform(0, 1, n)

    for i in range(n):
        x, y, b, t = int(xs[i]), int(ys[i]), bright[i], temps[i]
        r_ = np.clip(b*(0.86 + t*0.24 + max(0, color_temp)*0.07), 0, 1)
        g_ = np.clip(b*0.90, 0, 1)
        b_ = np.clip(b*(0.86 + (1-t)*0.28 + max(0, -color_temp)*0.09), 0, 1)
        img[y, x] = np.maximum(img[y, x], [r_, g_, b_])

    return img


def build_nebulas(nebulas, gamma=0.72, color_temp=0.0) -> np.ndarray:
    """
    Emission nebula: always-visible luminous clouds with filamentary texture.

    Architecture:
      base      = smooth envelope  → ensures nebula always glows
      texture   = turb5 warp       → adds filamentary ISM structure
      dark_lane = soft fbm detail  → carves thin absorption lanes (Hubble-style)
      combined  = additive blend on canvas with screen mode
    """
    gx, gy = np.meshgrid(
        np.linspace(0, 1, W, dtype=np.float32),
        np.linspace(0, 1, H, dtype=np.float32))

    rgb   = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W),    np.float32)

    for neb in nebulas:
        cx, cy = neb['cx'], neb['cy']
        rx, ry = neb.get('rx', 0.25), neb.get('ry', 0.22)
        col    = np.array(neb['color'], np.float32)
        inten  = neb['intensity']
        seed   = float(neb['seed'])

        dist     = np.sqrt(((gx-cx)/rx)**2 + ((gy-cy)/ry)**2)
        envelope = np.exp(-dist * dist * 1.8)   # softer falloff = wider glow
        active   = envelope > 0.008
        if not active.any(): continue

        fx, fy   = gx[active], gy[active]
        env_m    = envelope[active]

        # ── Domain warp (two rounds) for organic shape ───────────────────
        q0 = fbm5(fx*2.8 + seed*0.41+1.7, fy*2.8 + seed*0.13+9.2)
        q1 = fbm5(fx*2.8 + seed*0.17+8.3, fy*2.8 + seed*0.71+2.8)
        wux, wuy = fx*3.5 + q0*1.0, fy*3.5 + q1*1.0

        r0 = fbm5(wux + seed*0.23+4.1, wuy + 2.0)
        r1 = fbm5(wux + seed*0.59+5.1, wuy + 7.3)
        wux += r0*0.5; wuy += r1*0.5

        # ── Filamentary texture (turb = abs-fbm, creates wisps) ──────────
        texture = 0.25 + 0.75 * turb5(wux + seed*0.37, wuy + seed*0.19)

        # ── Soft dark absorption lanes (max 50% opacity reduction) ────────
        dark_raw  = fbm5(wux*1.6 + seed*0.5+3.3, wuy*1.6)
        dark_mod  = np.clip(0.50 + 0.50 * dark_raw, 0.45, 1.0)

        # ── Base always-on emission — proportional to envelope ────────────
        base = env_m * inten

        # ── Combine: base * texture * dark_mod, power < 1 brightens ──────
        a = np.power(np.clip(base * texture * dark_mod, 0, 1), gamma)

        # ── Emission: brighter near core (quadratic boost) ────────────────
        emission = 1.0 + 3.5 * env_m * env_m

        for c in range(3):
            rgb[:,:,c][active] = np.minimum(
                rgb[:,:,c][active] + col[c] * emission * a, 8.0)
        alpha[active] = np.maximum(alpha[active], np.clip(a * 1.1, 0, 1))

    # Filmic S-curve tonemapping
    rgb = rgb / (rgb + 0.35)
    rgb = np.power(np.clip(rgb, 0, 1), 0.85)

    # Color temperature tint
    warm = max(0, color_temp); cool = max(0, -color_temp)
    rgb[:,:,0] = np.clip(rgb[:,:,0] + warm*0.09, 0, 1)
    rgb[:,:,2] = np.clip(rgb[:,:,2] + cool*0.12, 0, 1)

    alpha = np.clip(alpha, 0, 1)
    return np.dstack([rgb, alpha[:,:,None]])


def draw_star(canvas: np.ndarray, px: int, py: int, mag: float,
              color: tuple, is_soul=False, glow_mult=1.0) -> None:
    """Multi-pass: outer halo → mid glow → diffraction spikes → bright core."""
    r        = 1.0 + (mag - 1) * 0.82
    glow_r   = r * (5.8 if is_soul else 4.0) * glow_mult
    spike_l  = r * (10 if is_soul else 7)
    cr, cg, cb = color

    def paint_glow(radius, alpha_center):
        """Paint a radial glow on the float canvas."""
        gr = int(radius * 3)
        x0, x1 = max(0, px-gr), min(W, px+gr+1)
        y0, y1 = max(0, py-gr), min(H, py+gr+1)
        ys_, xs_ = np.ogrid[y0:y1, x0:x1]
        d2 = (xs_ - px)**2 + (ys_ - py)**2
        mask = d2 < (radius*3)**2
        falloff = np.exp(-d2 / (radius**2 + 1e-9)) * alpha_center
        for c_i, c_v in enumerate([cr, cg, cb]):
            canvas[y0:y1, x0:x1, c_i] = np.clip(
                canvas[y0:y1, x0:x1, c_i] + c_v * falloff, 0, 1)

    # Outer glow
    paint_glow(glow_r,       0.18)
    # Mid glow
    paint_glow(glow_r * 0.45, 0.42)
    # Bright inner core
    paint_glow(r * 1.2,       0.85)

    # Diffraction spikes
    for deg in [0, 45, 90, 135]:
        rad = math.radians(deg)
        dx_f, dy_f = math.cos(rad), math.sin(rad)
        for t in np.linspace(-spike_l, spike_l, int(spike_l)*4):
            sx = int(px + dx_f * t); sy = int(py + dy_f * t)
            if 0 <= sx < W and 0 <= sy < H:
                fade = max(0, 1 - abs(t) / spike_l) * 0.55
                for c_i, c_v in enumerate([cr, cg, cb]):
                    canvas[sy, sx, c_i] = min(1, canvas[sy, sx, c_i] + c_v * fade)

    # Sharp white center
    rc = max(1, int(r))
    y0, y1 = max(0, py-rc), min(H, py+rc+1)
    x0, x1 = max(0, px-rc), min(W, px+rc+1)
    ys_, xs_ = np.ogrid[y0:y1, x0:x1]
    core = ((xs_-px)**2 + (ys_-py)**2) <= rc**2
    canvas[y0:y1, x0:x1, :3][core] = 1.0


def draw_constellation(canvas: np.ndarray) -> None:
    pts = [(int(x*W), int(y*H)) for (x, y) in CONSTELLATION]
    for (a, b) in CONST_EDGES:
        x1, y1 = pts[a]; x2, y2 = pts[b]
        # Bresenham line with alpha
        steps = max(abs(x2-x1), abs(y2-y1))
        for t in np.linspace(0, 1, steps*2):
            lx = int(x1 + (x2-x1)*t); ly = int(y1 + (y2-y1)*t)
            if 0 <= lx < W and 0 <= ly < H:
                canvas[ly, lx, :3] = np.minimum(
                    canvas[ly, lx, :3] + np.array([0.63, 0.70, 0.86])*0.45, 1)


def add_vignette(canvas: np.ndarray, strength=0.5) -> None:
    gx, gy = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
    v = np.clip(1 - strength*(gx**2 + gy**2)*1.6, 0, 1)[:,:,None]
    canvas[:,:,:3] *= v


def add_labels(pil_img: Image.Image) -> None:
    try:
        from PIL import ImageFont
        font    = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)
        font_sm = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 15)
    except Exception:
        from PIL import ImageFont
        font = font_sm = ImageFont.load_default()

    d = ImageDraw.Draw(pil_img)
    for star in SURFACE_STARS:
        px, py = int(star['x']*W), int(star['y']*H)
        cr, cg, cb = (int(c*255) for c in star['color'])
        off = 22 if star.get('soul') else 16
        d.text((px+off, py-10), star['label'], fill=(cr,cg,cb,215), font=font_sm)

    d.text((W//2-100, 28), "哈利·波特  灵魂星图",
           fill=(180,200,225,170), font=font)
    d.text((W//2-110, H-42), "孤独 · 使命 · 悲伤 · 当下紧迫感",
           fill=(120,140,175,130), font=font_sm)


# ─── Main render ─────────────────────────────────────────────────────────────

def render(out_path: str) -> None:
    p = PARAMS
    print("Rendering soul sky …")

    # 1. Starfield
    print("  [1/5] Starfield …")
    canvas = build_starfield(density=p['bgDensity'], color_temp=p['colorTemp'])

    # 2. Nebula
    print("  [2/5] Nebula (domain-warp, 5-oct turbulence) …")
    neb = build_nebulas(NEBULAS, gamma=p['nebulaGamma'], color_temp=p['colorTemp'])
    neb_rgb, neb_a = neb[:,:,:3], neb[:,:,3:4]

    # Screen blend nebula onto starfield
    canvas = np.clip(1 - (1 - canvas)*(1 - neb_rgb*neb_a), 0, 1)

    # 3. Vignette
    add_vignette(canvas, p['vigStrength'])

    # 4. Constellations
    print("  [3/5] Constellations + stars …")
    draw_constellation(canvas)

    # 5. Surface stars
    for star in sorted(SURFACE_STARS, key=lambda s: s['mag']):
        draw_star(canvas,
                  int(star['x']*W), int(star['y']*H),
                  mag=star['mag'], color=star['color'],
                  is_soul=star.get('soul', False),
                  glow_mult=p['soulGlow'] if star.get('soul') else 1.0)

    # Labels
    print("  [4/5] Labels …")
    pil_img = Image.fromarray((canvas*255).clip(0,255).astype(np.uint8))
    add_labels(pil_img)

    out = Path(out_path).expanduser()
    pil_img.save(out)
    print(f"\n  ✓  {out}  ({W}×{H})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='~/Desktop/soul_sky_preview.png')
    render(ap.parse_args().out)
