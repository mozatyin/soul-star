#!/usr/bin/env python3
"""
soul_star._cinema — Cinematic 1920×1080 compositing layer.

Takes individual soul star frames (910×910 or any size PNG) and composites
them into a widescreen cinema canvas with deep-space side panels, parallax
stars, nebula blobs, lens flare, vignette, film grain, and color grading.
"""

import math
import numpy as np
from PIL import Image, ImageEnhance

# ── Module-level constants ─────────────────────────────────────────────────
CINEMA_P = {
    'cinema_w': 1920,
    'cinema_h': 1080,
    'soul_fig_sz': 10.8,        # figsize for soul star → 1080×1080 at DPI=100
    'depth_layers': 5,
    'depth_star_counts': [301, 170, 92, 40],     # stars per layer — sparse, background serves foreground
    'depth_alphas': [0.0548, 0.1148, 0.2227, 0.4805],
    'depth_radii': [0.583, 1.239, 2.252, 3.665],         # pixel radius per layer
    'parallax_amp': 0.0,        # LOCKED: background stars absolutely static — twinkle only, never move
    'lens_flare_intensity': 0.504,
    'vignette_strength': 0.707,
    'color_grade_temp': -0.029,
    'color_grade_saturation': 1.164,
    'side_nebula_alpha': 0.237,
    'side_aurora_mult': 1.395,
    'film_grain': 0.017,
}


# ── Private helpers ────────────────────────────────────────────────────────

def _draw_depth_stars(canvas_arr, x_offset, panel_w, panel_h, frame_idx,
                      n_frames, cinema_p, rng_seed):
    """
    Draw multi-layer parallax stars into canvas_arr in the panel region
    [0:panel_h, x_offset:x_offset+panel_w].

    Each layer uses a deterministic RNG (rng_seed + layer_i*31).
    Far layers (layer 0) drift less; near layers (layer 3) drift more.
    Star colors are slightly blue-tinted (B channel boosted 10%).

    Parameters
    ----------
    canvas_arr : np.ndarray  shape (H, W, 3) uint8, modified in-place
    x_offset   : int         left edge of panel in canvas
    panel_w    : int         panel width in pixels
    panel_h    : int         panel height in pixels
    frame_idx  : int         current frame index (0-based)
    n_frames   : int         total frames in animation
    cinema_p   : dict        CINEMA_P parameters
    rng_seed   : int         base RNG seed for this panel
    """
    star_counts = cinema_p['depth_star_counts']
    alphas = cinema_p['depth_alphas']
    radii = cinema_p['depth_radii']
    # Clamp to minimum array length to guard against director-adjusted mismatches
    n_layers = min(cinema_p['depth_layers'], len(star_counts), len(alphas), len(radii))
    parallax_amp = cinema_p['parallax_amp']
    panel_h_full = cinema_p['cinema_h']

    # Animation phase: slow oscillation across the animation
    phase = (frame_idx / max(n_frames - 1, 1)) * 2.0 * math.pi

    for layer_i in range(n_layers):
        rng = np.random.RandomState(rng_seed + layer_i * 31)
        n_stars = star_counts[layer_i]
        alpha = alphas[layer_i]
        radius = radii[layer_i]

        # Parallax drift: layer 0 (far) = slowest, layer 3 (near) = fastest
        # drift_factor ranges from 0.25 (far) to 1.0 (near)
        drift_factor = (layer_i + 1) / n_layers
        drift_x = drift_factor * parallax_amp * panel_w * math.sin(phase)
        drift_y = drift_factor * parallax_amp * panel_h_full * math.cos(phase * 0.7)

        # Generate star base positions within the panel
        base_xs = rng.uniform(0, panel_w, n_stars)
        base_ys = rng.uniform(0, panel_h, n_stars)

        # Base star colors: slightly blue-tinted whites/grays
        base_colors = rng.uniform(0.7, 1.0, (n_stars, 3))
        # Boost blue channel 10%
        base_colors[:, 2] = np.clip(base_colors[:, 2] * 1.10, 0.0, 1.0)
        # Slight variation: some stars more blue, some more neutral
        hue_var = rng.uniform(-0.05, 0.15, n_stars)
        base_colors[:, 2] = np.clip(base_colors[:, 2] + hue_var, 0.0, 1.0)

        int_radius = int(math.ceil(radius))

        for s_i in range(n_stars):
            # Apply parallax drift
            sx = base_xs[s_i] + drift_x
            sy = base_ys[s_i] + drift_y

            # Wrap within panel (toroidal)
            sx = sx % panel_w
            sy = sy % panel_h

            cx = int(round(sx)) + x_offset
            cy = int(round(sy))

            r_val = base_colors[s_i, 0]
            g_val = base_colors[s_i, 1]
            b_val = base_colors[s_i, 2]

            # Paint star disk with gaussian-like falloff
            for dy in range(-int_radius, int_radius + 1):
                py = cy + dy
                if py < 0 or py >= panel_h:
                    continue
                for dx in range(-int_radius, int_radius + 1):
                    px = cx + dx
                    if px < x_offset or px >= x_offset + panel_w:
                        continue
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > radius + 0.5:
                        continue
                    # Gaussian falloff
                    falloff = math.exp(-dist * dist / (radius * radius + 1e-9))
                    blend = alpha * falloff
                    existing = canvas_arr[py, px]
                    canvas_arr[py, px, 0] = int(min(255, existing[0] + blend * r_val * 255))
                    canvas_arr[py, px, 1] = int(min(255, existing[1] + blend * g_val * 255))
                    canvas_arr[py, px, 2] = int(min(255, existing[2] + blend * b_val * 255))


def _make_nebula_overlay(panel_w, panel_h, aurora_val, cinema_p, rng_seed):
    """Return float32 (H, W, 3) nebula overlay in [0, 1]. Vectorized."""
    alpha_scale = cinema_p['side_nebula_alpha'] * cinema_p['side_aurora_mult'] * aurora_val
    if alpha_scale < 0.01:
        return np.zeros((panel_h, panel_w, 3), dtype=np.float32)

    rng = np.random.RandomState(rng_seed + 99)
    n_blobs = rng.randint(1, 3)
    overlay = np.zeros((panel_h, panel_w, 3), dtype=np.float32)
    Y, X = np.mgrid[0:panel_h, 0:panel_w]

    for _ in range(n_blobs):
        cx = rng.uniform(0.2, 0.8) * panel_w
        cy = rng.uniform(0.2, 0.8) * panel_h
        rx = rng.uniform(0.15, 0.4) * panel_w
        ry = rng.uniform(0.1, 0.3) * panel_h
        col = np.array([
            rng.uniform(0.3, 0.6),
            rng.uniform(0.4, 0.8),
            rng.uniform(0.6, 1.0)
        ], dtype=np.float32)
        d2 = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask = d2 < 1.0
        intensity = np.where(mask, alpha_scale * (1 - d2), 0).astype(np.float32)
        overlay += intensity[:, :, np.newaxis] * col

    return np.clip(overlay, 0, 1)


def _apply_vignette(img_arr, strength):
    """
    Darken edges of a full-frame numpy array using a radial vignette.

    Parameters
    ----------
    img_arr  : np.ndarray  shape (H, W, 3) uint8
    strength : float       vignette strength [0..1], 0 = no effect

    Returns
    -------
    np.ndarray  shape (H, W, 3) uint8
    """
    h, w = img_arr.shape[:2]
    # Normalized coordinates [-1, 1]
    xs = np.linspace(-1.0, 1.0, w)
    ys = np.linspace(-1.0, 1.0, h)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)

    # Vignette mask: 1 at center, falling off at edges
    # dist ranges from 0 (center) to ~1.41 (corners)
    vignette = 1.0 - strength * np.clip(dist / 1.41, 0.0, 1.0) ** 1.5
    vignette = np.clip(vignette, 0.0, 1.0).astype(np.float32)

    # Apply to all channels
    out = img_arr.astype(np.float32)
    out[:, :, 0] *= vignette
    out[:, :, 1] *= vignette
    out[:, :, 2] *= vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_color_grade(pil_img, temp_shift, saturation):
    """
    Apply color grading: saturation enhancement + temperature channel shift.

    temp_shift < 0 → cooler/bluer:
        boost blue by abs(temp)*0.3, reduce red by abs(temp)*0.15
    temp_shift > 0 → warmer:
        boost red by temp*0.2, green by temp*0.1, reduce blue by temp*0.1

    Parameters
    ----------
    pil_img    : PIL.Image  RGB image
    temp_shift : float      color temperature shift
    saturation : float      saturation multiplier (1.0 = unchanged)

    Returns
    -------
    PIL.Image  RGB image
    """
    # Apply saturation
    img = ImageEnhance.Color(pil_img).enhance(saturation)

    # Channel-level temperature shift
    arr = np.array(img, dtype=np.float32)

    if temp_shift < 0:
        t = abs(temp_shift)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 - t * 0.15), 0, 255)  # reduce red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 + t * 0.30), 0, 255)  # boost blue
    elif temp_shift > 0:
        t = temp_shift
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + t * 0.20), 0, 255)  # boost red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1.0 + t * 0.10), 0, 255)  # boost green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - t * 0.10), 0, 255)  # reduce blue

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='RGB')


def _apply_film_grain(arr, strength, rng_seed):
    """
    Add film grain noise to a numpy array.

    Parameters
    ----------
    arr      : np.ndarray  shape (H, W, 3) uint8
    strength : float       grain strength (e.g. 0.012)
    rng_seed : int         RNG seed for reproducibility

    Returns
    -------
    np.ndarray  shape (H, W, 3) uint8
    """
    rng = np.random.RandomState(rng_seed)
    noise = rng.normal(0.0, strength * 255, arr.shape[:2])
    out = arr.astype(np.float32)
    # Apply same noise to all channels
    out[:, :, 0] += noise
    out[:, :, 1] += noise
    out[:, :, 2] += noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Main compositing function ──────────────────────────────────────────────

def composite_cinema_frame(soul_frame_path, frame_idx, n_frames,
                           cinema_p, bg_temp, aurora_val):
    """
    Composite a soul star frame into a 1920×1080 cinematic canvas.

    Layout
    ------
    [  left panel 420px  |  soul star 1080px  |  right panel 420px  ]
    x=0..419               x=420..1499          x=1500..1919

    Parameters
    ----------
    soul_frame_path : str | Path   path to the soul star PNG frame
    frame_idx       : int          current frame index (0-based)
    n_frames        : int          total frames in animation
    cinema_p        : dict         CINEMA_P parameters dict
    bg_temp         : float        background temperature [-1..1]
    aurora_val      : float        aurora intensity [0..1]

    Returns
    -------
    PIL.Image  1920×1080 RGB image
    """
    cinema_w = cinema_p['cinema_w']
    cinema_h = cinema_p['cinema_h']
    soul_w = cinema_h  # 1080
    soul_h = cinema_h  # 1080
    panel_w = (cinema_w - soul_w) // 2   # 420
    soul_x = panel_w                      # 420

    # 1. Allocate black canvas
    canvas = np.zeros((cinema_h, cinema_w, 3), dtype=np.uint8)

    # 2. Draw depth stars uniformly across the full 1920px canvas.
    #    Scale star counts proportionally so density matches the original side panels.
    #    Soul star is alpha-composited on top, so its dark background becomes transparent
    #    and cinema stars are visible everywhere with no hard boundary.
    _scale = cinema_w / panel_w   # 1920 / 420 ≈ 4.57
    _full_cp = dict(cinema_p)
    _full_cp['depth_star_counts'] = [int(round(c * _scale)) for c in cinema_p['depth_star_counts']]
    _draw_depth_stars(
        canvas, x_offset=0, panel_w=cinema_w, panel_h=cinema_h,
        frame_idx=frame_idx, n_frames=n_frames, cinema_p=_full_cp,
        rng_seed=42
    )

    # 3. Nebula overlays — side panels full intensity, faint across center for continuity
    canvas_float = canvas.astype(np.float32)
    neb_left = _make_nebula_overlay(panel_w, cinema_h, aurora_val, cinema_p, 42)
    canvas_float[:, :panel_w] = np.clip(
        canvas_float[:, :panel_w] + neb_left * 255, 0, 255
    )
    neb_right = _make_nebula_overlay(panel_w, cinema_h, aurora_val, cinema_p, 137 + 1000)
    canvas_float[:, soul_x + soul_w:] = np.clip(
        canvas_float[:, soul_x + soul_w:] + neb_right * 255, 0, 255
    )
    neb_center = _make_nebula_overlay(soul_w, cinema_h, aurora_val * 0.4, cinema_p, 999)
    canvas_float[:, soul_x:soul_x + soul_w] = np.clip(
        canvas_float[:, soul_x:soul_x + soul_w] + neb_center * 255, 0, 255
    )
    canvas = canvas_float.astype(np.uint8)

    # 4. Screen-blend soul star over cinema background.
    #    Screen formula: result = 1 - (1-soul)*(1-bg)
    #    → near-black soul pixels ≈ 0 contribute nothing (background shows through)
    #    → bright nebulas/nodes/labels stack luminously on top of cinema stars
    #    No boundary lines, no chroma-key artefacts.
    soul_img = Image.open(soul_frame_path).convert('RGB')
    soul_img = soul_img.resize((soul_w, soul_h), Image.LANCZOS)
    soul_norm = np.array(soul_img, dtype=np.float32) / 255.0   # (H, W, 3) in [0,1]

    canvas_norm = canvas.astype(np.float32) / 255.0
    bg_region = canvas_norm[:, soul_x:soul_x + soul_w, :]
    # Screen blend
    screened = 1.0 - (1.0 - soul_norm) * (1.0 - bg_region)
    canvas_norm[:, soul_x:soul_x + soul_w, :] = screened
    canvas = np.clip(canvas_norm * 255, 0, 255).astype(np.uint8)

    # 5. Apply vignette to full frame
    canvas = _apply_vignette(canvas, cinema_p['vignette_strength'])

    # 6. Apply film grain (seed based on frame_idx for frame-to-frame variation)
    canvas = _apply_film_grain(canvas, cinema_p['film_grain'],
                               rng_seed=frame_idx + 7777)

    # 7. Convert to PIL Image and apply color grade
    pil_frame = Image.fromarray(canvas, mode='RGB')
    pil_frame = _apply_color_grade(
        pil_frame,
        temp_shift=cinema_p['color_grade_temp'],
        saturation=cinema_p['color_grade_saturation'],
    )

    # 8. Return final 1920×1080 PIL Image RGB
    assert pil_frame.size == (cinema_w, cinema_h), (
        f"Unexpected output size: {pil_frame.size}, expected ({cinema_w}, {cinema_h})")
    return pil_frame
