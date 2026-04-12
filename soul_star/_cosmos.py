"""
soul_star._cosmos — Cosmos View renderer.

Each soul element (relation, ecology) performs as its assigned cosmic phenomenon
instead of a generic star.  Domain nebulae stay unchanged.

Cosmos types and their visual logic
────────────────────────────────────
comet           — slow sweeping bright head + fading tail, dark ominous tint
supernova       — point that builds to a blinding flash, then fades to a ring
globular_cluster— dense tight swarm of micro-stars
meteor          — periodic bright streak passing through the element's position
grav_ripple     — concentric expanding rings (invisible force bending light)
satellite       — slow steady dot tracing an arc through the scene
dark_cloud      — soft darkening blob that breathes with intensity
aurora_surge    — colored radial flare burst
variable_star   — breathing point-source with sinusoidal brightness

All renderers share the same signature:
    render_cosmos_element(ax, cx, cy, intensity, frame_idx, n_frames,
                          cosmos_type, color_hex, rng_seed, fig_sz, W, H)

    ax         : matplotlib Axes (world coords 0..W × 0..H)
    cx, cy     : world-coord centre of this element
    intensity  : 0–1  (lerp(curve, pct) / 10.0)
    frame_idx  : current frame (0-based)
    n_frames   : total frames in animation
    cosmos_type: str — one of the 9 types above
    color_hex  : str — element's characteristic colour
    rng_seed   : int — deterministic seed for this element
    fig_sz     : float — figsize for size scaling (same as render_frame)
    W, H       : float — world canvas dimensions
"""

import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Internal helpers ──────────────────────────────────────────────────────────

def _pts(fs, W):
    """Points-per-world-unit (axes DPI helper, identical to _engine._pts)."""
    return fs * 100 / W   # DPI=100 assumed

def _s(r, fs, W):
    return max(0.1, (r * _pts(fs, W) * 2.0) ** 2)

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _lerp_color(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))

def _fmt(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# ── Per-type renderers ────────────────────────────────────────────────────────

def _comet(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Slow dark comet sweeping across; head at (cx,cy), tail trails behind."""
    if intensity < 0.05:
        return
    rng = np.random.RandomState(rng_seed)
    phase = frame_idx / max(n_frames - 1, 1)  # 0→1

    # Slow oscillating drift around anchor position
    drift_x = math.sin(phase * math.pi * 1.5 + rng.uniform(0, math.pi)) * 2.0 * intensity
    drift_y = math.cos(phase * math.pi * 1.2 + rng.uniform(0, math.pi)) * 1.2 * intensity
    hx = cx + drift_x
    hy = cy + drift_y

    col = _hex_to_rgb(color_hex)
    # Tail: 8 segments trailing opposite to drift direction
    tail_len = 3.5 * intensity
    tail_dir_x = -drift_x / max(abs(drift_x) + 1e-6, abs(drift_y) + 1e-6)
    tail_dir_y = -drift_y / max(abs(drift_x) + 1e-6, abs(drift_y) + 1e-6)
    n_seg = 10
    for i in range(n_seg):
        t = i / n_seg
        sx = hx + tail_dir_x * tail_len * t
        sy = hy + tail_dir_y * tail_len * t
        alpha = intensity * (1.0 - t) * 0.65
        r_tail = (0.20 - 0.15 * t) * intensity
        ax.scatter([sx], [sy], s=_s(r_tail, fig_sz, W),
                   c=[col], alpha=float(np.clip(alpha, 0, 1)),
                   linewidths=0, zorder=6)

    # Bright head
    head_r = 0.28 * intensity
    ax.scatter([hx], [hy], s=_s(head_r * 2.5, fig_sz, W),
               c=['white'], alpha=float(intensity * 0.9), linewidths=0, zorder=6.5)
    ax.scatter([hx], [hy], s=_s(head_r, fig_sz, W),
               c=[col], alpha=float(intensity * 0.7), linewidths=0, zorder=6.5)


def _supernova(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Builds to a blinding flash, then leaves an expanding remnant ring."""
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)
    phase = frame_idx / max(n_frames - 1, 1)

    # Classify: pre-flash (steady glow), flash (brief bright peak), post-flash (expanding ring)
    # Flash is keyed to when intensity peaks in the animation — approximate with a simple model
    # Use a synthetic brightness envelope: rises with intensity, has a spike at high intensity
    brightness = intensity ** 1.2

    # Glow halo — always present, scales with intensity
    glow_r = 0.6 + brightness * 1.2
    alpha_glow = 0.18 + brightness * 0.35
    ax.scatter([cx], [cy], s=_s(glow_r * 4.0, fig_sz, W),
               c=[col], alpha=float(np.clip(alpha_glow, 0, 0.85)), linewidths=0, zorder=5)

    # Core bright point
    core_r = 0.15 + brightness * 0.40
    ax.scatter([cx], [cy], s=_s(core_r * 2.0, fig_sz, W),
               c=['white'], alpha=float(np.clip(brightness * 0.95, 0, 1)), linewidths=0, zorder=5.5)
    ax.scatter([cx], [cy], s=_s(core_r, fig_sz, W),
               c=[col], alpha=float(np.clip(brightness * 0.80, 0, 1)), linewidths=0, zorder=5.5)

    # Expanding remnant ring (visible after peak intensity)
    if intensity > 0.4:
        ring_r = 0.5 + (1.0 - intensity) * 2.5   # expands as intensity falls post-peak
        ring_alpha = max(0.0, (intensity - 0.4) / 0.6 * 0.45)
        ring_col = _lerp_color(col, (1.0, 1.0, 1.0), 0.5)
        ring = mpatches.Circle((cx, cy), ring_r, fill=False,
                                edgecolor=_fmt(ring_col),
                                linewidth=max(0.5, 1.5 * intensity),
                                alpha=float(ring_alpha), zorder=5.2)
        ax.add_patch(ring)


def _globular_cluster(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Dense tight swarm of micro-stars centred at (cx,cy)."""
    if intensity < 0.05:
        return
    rng = np.random.RandomState(rng_seed)
    col = _hex_to_rgb(color_hex)
    n = int(30 + intensity * 120)
    spread = 0.8 + intensity * 1.5
    r_vals = np.abs(rng.normal(0, spread * 0.4, n))
    a_vals = rng.uniform(0, 2 * math.pi, n)
    xs = cx + r_vals * np.cos(a_vals)
    ys = cy + r_vals * np.sin(a_vals)
    mags = rng.uniform(0.04, 0.18, n) * intensity
    alphas = np.clip(mags / 0.18 * 0.80, 0.05, 0.80)

    # Slight twinkle tied to frame
    tw = 0.85 + 0.15 * math.sin(frame_idx * 0.4 + rng_seed)

    for i in range(n):
        ax.scatter([float(xs[i])], [float(ys[i])],
                   s=_s(float(mags[i]) * tw, fig_sz, W),
                   c=[col], alpha=float(alphas[i]),
                   linewidths=0, zorder=5)

    # Bright centre condensation
    ax.scatter([cx], [cy], s=_s(0.22 * intensity, fig_sz, W),
               c=['white'], alpha=float(intensity * 0.85), linewidths=0, zorder=5.5)


def _meteor(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Periodic bright streak passing near (cx,cy)."""
    if intensity < 0.1:
        return
    rng = np.random.RandomState(rng_seed)
    col = _hex_to_rgb(color_hex)

    # Fire a streak every ~20 frames; position jitter each time
    period = max(8, int(20 / max(intensity, 0.1)))
    local = frame_idx % period
    if local > 6:
        return   # only visible for first 6 frames of each period

    streak_phase = local / 6.0   # 0→1 within streak
    rng2 = np.random.RandomState(rng_seed + frame_idx // period)
    angle = rng2.uniform(0.15, math.pi - 0.15)
    length = 1.5 + intensity * 2.5
    # Start offset from anchor
    ox = cx + rng2.uniform(-1.5, 1.5)
    oy = cy + rng2.uniform(-1.5, 1.5)
    x0 = ox - math.cos(angle) * length * 0.5
    y0 = oy - math.sin(angle) * length * 0.5
    x1 = ox + math.cos(angle) * length * 0.5
    y1 = oy + math.sin(angle) * length * 0.5

    # Head position
    hx = x0 + (x1 - x0) * streak_phase
    hy = y0 + (y1 - y0) * streak_phase
    fade = (1.0 - streak_phase) ** 0.5
    bright_col = _lerp_color(col, (1.0, 1.0, 1.0), 0.6)

    n_seg = 8
    for i in range(n_seg):
        t = i / n_seg
        sx = x0 + (x1 - x0) * streak_phase * t
        sy = y0 + (y1 - y0) * streak_phase * t
        alpha = intensity * (1.0 - t) * fade * 0.75
        r = (0.18 - 0.14 * t) * intensity
        ax.scatter([sx], [sy], s=_s(r, fig_sz, W),
                   c=[bright_col], alpha=float(np.clip(alpha, 0, 1)),
                   linewidths=0, zorder=6)


def _grav_ripple(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Concentric expanding rings — invisible force bending light."""
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)
    # Multiple rings, each offset by a phase so they look continuously expanding
    n_rings = 3
    for i in range(n_rings):
        ring_phase = (frame_idx / max(n_frames - 1, 1) + i / n_rings) % 1.0
        ring_r = 0.3 + ring_phase * (1.5 + intensity * 2.5)
        ring_alpha = intensity * (1.0 - ring_phase) * 0.55
        if ring_alpha < 0.02:
            continue
        ring = mpatches.Circle((cx, cy), ring_r, fill=False,
                                edgecolor=_fmt(col),
                                linewidth=max(0.4, 1.2 * intensity * (1 - ring_phase)),
                                alpha=float(ring_alpha), zorder=4.8)
        ax.add_patch(ring)

    # Faint centre pulse
    pulse = 0.5 + 0.5 * math.sin(frame_idx * 0.25)
    ax.scatter([cx], [cy], s=_s(0.12 * intensity * pulse, fig_sz, W),
               c=[col], alpha=float(intensity * 0.50), linewidths=0, zorder=5)


def _satellite(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Slow faithful dot tracing a gentle arc through the scene."""
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)
    # Arc: slow ellipse orbit around (cx, cy)
    orbit_rx = 1.8 + intensity * 2.0
    orbit_ry = 0.9 + intensity * 1.2
    orbit_phase = (frame_idx / max(n_frames - 1, 1)) * 2 * math.pi
    sx = cx + orbit_rx * math.cos(orbit_phase)
    sy = cy + orbit_ry * math.sin(orbit_phase * 0.7)  # non-circular for interest
    sx = float(np.clip(sx, 0.5, W - 0.5))
    sy = float(np.clip(sy, 0.5, H - 0.5))

    bright = _lerp_color(col, (1.0, 1.0, 1.0), 0.5)
    ax.scatter([sx], [sy], s=_s(0.18 * intensity, fig_sz, W),
               c=[bright], alpha=float(intensity * 0.90), linewidths=0, zorder=6)
    # Soft cross-glow
    ax.scatter([sx], [sy], s=_s(0.35 * intensity, fig_sz, W),
               c=[col], alpha=float(intensity * 0.35), linewidths=0, zorder=5.9)


def _dark_cloud(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Soft darkening blob that breathes with intensity."""
    if intensity < 0.05:
        return
    rng = np.random.RandomState(rng_seed)
    # Breathing pulse
    breath = 1.0 + 0.08 * math.sin(frame_idx * 0.18)
    cloud_r_x = (2.0 + intensity * 2.5) * breath
    cloud_r_y = (1.4 + intensity * 1.8) * breath
    opacity = 0.18 + intensity * 0.40

    # Draw as a dark ellipse overlay
    dark = mpatches.Ellipse((cx, cy), cloud_r_x * 2, cloud_r_y * 2,
                             fill=True, facecolor='#000000',
                             edgecolor='none', alpha=float(np.clip(opacity, 0, 0.65)),
                             zorder=3.5)
    ax.add_patch(dark)

    # Faint green-tinted edge glow (horcrux corruption)
    col = _hex_to_rgb(color_hex)
    edge = mpatches.Ellipse((cx, cy), cloud_r_x * 2.3, cloud_r_y * 2.3,
                             fill=False, edgecolor=_fmt(col),
                             linewidth=max(0.5, 1.5 * intensity),
                             alpha=float(intensity * 0.30), zorder=3.6)
    ax.add_patch(edge)


def _aurora_surge(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Coloured radial flare burst centred at (cx,cy)."""
    if intensity < 0.05:
        return
    rng = np.random.RandomState(rng_seed)
    col = _hex_to_rgb(color_hex)
    white = (1.0, 1.0, 1.0)

    # Pulse phase
    pulse = 0.6 + 0.4 * math.sin(frame_idx * 0.35)

    # Outer halo
    halo_r = (1.2 + intensity * 2.2) * pulse
    halo_col = _lerp_color(col, white, 0.3)
    ax.scatter([cx], [cy], s=_s(halo_r * 5.0, fig_sz, W),
               c=[halo_col], alpha=float(intensity * 0.22 * pulse),
               linewidths=0, zorder=4.5)

    # Mid ring
    mid_r = halo_r * 0.55
    ax.scatter([cx], [cy], s=_s(mid_r * 3.0, fig_sz, W),
               c=[col], alpha=float(intensity * 0.45 * pulse),
               linewidths=0, zorder=4.6)

    # Bright core
    ax.scatter([cx], [cy], s=_s(0.20 * intensity, fig_sz, W),
               c=['white'], alpha=float(intensity * 0.90),
               linewidths=0, zorder=4.7)

    # Ray spokes (4 directions)
    spoke_len = 0.6 + intensity * 1.2
    for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
        rad = math.radians(angle_deg)
        ex = cx + math.cos(rad) * spoke_len * pulse
        ey = cy + math.sin(rad) * spoke_len * pulse
        ax.plot([cx, ex], [cy, ey], '-', color=_fmt(col),
                alpha=float(intensity * 0.30 * pulse),
                linewidth=max(0.3, 1.0 * intensity),
                solid_capstyle='round', zorder=4.5)


def _variable_star(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """Breathing point-source with sinusoidal brightness — mysterious pulsation."""
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)
    # Long-period pulse (slower = more mysterious)
    period = max(n_frames // 4, 8)
    pulse = 0.4 + 0.6 * math.sin(2 * math.pi * frame_idx / period)
    brightness = intensity * pulse

    # Outer aura
    aura_r = 0.4 + brightness * 1.8
    ax.scatter([cx], [cy], s=_s(aura_r * 4.0, fig_sz, W),
               c=[col], alpha=float(brightness * 0.28),
               linewidths=0, zorder=5)

    # Core star
    core_r = 0.10 + brightness * 0.32
    ax.scatter([cx], [cy], s=_s(core_r * 2.5, fig_sz, W),
               c=['white'], alpha=float(np.clip(brightness * 0.95, 0, 1)),
               linewidths=0, zorder=5.5)
    ax.scatter([cx], [cy], s=_s(core_r, fig_sz, W),
               c=[col], alpha=float(np.clip(brightness * 0.75, 0, 1)),
               linewidths=0, zorder=5.5)

    # 4-point diffraction spike (variable stars show these in telescopes)
    spike_len = core_r * 6.0 * brightness
    for angle_deg in (0, 90, 180, 270):
        rad = math.radians(angle_deg)
        ax.plot([cx, cx + math.cos(rad) * spike_len],
                [cy, cy + math.sin(rad) * spike_len],
                '-', color=_fmt(col),
                alpha=float(brightness * 0.55),
                linewidth=max(0.3, 0.8 * brightness),
                solid_capstyle='round', zorder=5.3)


# ── Dispatch table ────────────────────────────────────────────────────────────

_RENDERERS = {
    'comet':             _comet,
    'supernova':         _supernova,
    'globular_cluster':  _globular_cluster,
    'meteor':            _meteor,
    'grav_ripple':       _grav_ripple,
    'satellite':         _satellite,
    'dark_cloud':        _dark_cloud,
    'aurora_surge':      _aurora_surge,
    'variable_star':     _variable_star,
}

# ── Public API ────────────────────────────────────────────────────────────────

def render_cosmos_element(ax, cx, cy, intensity, frame_idx, n_frames,
                          cosmos_type, color_hex, rng_seed, fig_sz, W, H):
    """
    Render one soul element as its assigned cosmic phenomenon.

    Parameters
    ----------
    ax          : matplotlib Axes
    cx, cy      : float  world-coordinate centre of this element
    intensity   : float  0–1 (curve value normalised)
    frame_idx   : int
    n_frames    : int
    cosmos_type : str    one of the 9 keys in _RENDERERS
    color_hex   : str    e.g. '#30b050'
    rng_seed    : int    deterministic per-element seed
    fig_sz      : float  figsize (same as render_frame)
    W, H        : float  world canvas size
    """
    fn = _RENDERERS.get(cosmos_type)
    if fn is None:
        return   # unknown type — silently skip
    fn(ax, cx, cy, float(np.clip(intensity, 0, 1)), frame_idx, n_frames,
       color_hex, rng_seed, fig_sz, W, H)


def cosmos_label(ax, cx, cy, label, color_hex, cosmos_type, fig_sz, W, H):
    """
    Draw a small italic label near the cosmos element.
    Positioned above/below to avoid overlapping the phenomenon.
    """
    offset = {'comet': 0.9, 'supernova': 1.1, 'globular_cluster': 1.2,
              'meteor': 0.7, 'grav_ripple': 0.7, 'satellite': 0.5,
              'dark_cloud': 1.0, 'aurora_surge': 0.9, 'variable_star': 0.7}
    off = offset.get(cosmos_type, 0.8)
    ty = min(cy + off, H - 0.3)
    ax.text(cx, ty, label, fontsize=9.0, color=color_hex, alpha=0.72,
            ha='center', va='bottom', fontstyle='italic', fontweight='light',
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground='#000000')])
