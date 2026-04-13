"""
soul_star._cosmos — Cosmos View renderer (v2 — numpy pixel-buffer edition).

All renderers use float32 RGBA pixel buffers with additive-safe compositing
instead of matplotlib scatter, eliminating alpha > 1.0 crashes and enabling
true additive blending for stars, glows, and energy phenomena.

Cosmos types and canonical algorithms:
  comet           — dual-tail capsule SDF (dust warm-curved, ion blue-straight)
                    + 3-layer Gaussian coma + Moffat PSF nucleus
  supernova       — 3-phase timeline: compact pre-flash / white bloom /
                    Sedov-Taylor limb-brightened shell + fBm ejecta filaments
  globular_cluster— King (1962) density field + rejection-sampled discrete stars
                    rendered with per-star Moffat PSF
  meteor          — tapered capsule SDF + ablation colour gradient (white→col)
                    + luminous Moffat head
  grav_ripple     — quadrupolar cos²(2θ) concentric rings, 1/r amplitude falloff
  satellite       — Keplerian orbit (Newton solver) + fading dot trail
                    + solar glint cross-spike
  dark_cloud      — fBm domain-warped Beer-Lambert absorption nebula
                    + faint element-colour edge glow
  aurora_surge    — triangle-wave curtain columns + altitude-layered colour
                    (purple N₂ → green O → red O⁺) + gaussian blur
  variable_star   — Fourier Cepheid light curve + Mitchell blackbody T-shift
                    + Moffat PSF + 4-point diffraction spikes
"""

import math
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pe

# ── Pixel-buffer infrastructure ───────────────────────────────────────────────

def _ppu(fs, W):
    """Pixels per world unit.  fig_sz × DPI(100) / world_width."""
    return fs * 100.0 / W


def _patch(cx, cy, r_w, fs, W, H, min_n=32, max_n=224):
    """
    Allocate a square float32 RGBA pixel buffer for a circular patch.
    Returns (buf[N,N,4], extent=[x0,x1,y0,y1], N).
    """
    ppu = _ppu(fs, W)
    N = min(max_n, max(min_n, int(r_w * 2.0 * ppu)))
    N += N % 2  # keep even
    return (np.zeros((N, N, 4), dtype=np.float32),
            [cx - r_w, cx + r_w, cy - r_w, cy + r_w],
            N)


def _coords(N, extent):
    """
    Return (X, Y) world-coord meshgrids for an (N, N) buffer.
    origin='lower': row-0 = y0, row-(N-1) = y1.
    """
    x0, x1, y0, y1 = extent
    X, Y = np.meshgrid(
        np.linspace(x0, x1, N, dtype=np.float32),
        np.linspace(y0, y1, N, dtype=np.float32),
    )
    return X, Y


def _show(ax, buf, extent, zorder=5):
    """
    Display pixel buffer on axes.
    Alpha = max(R,G,B) × 1.5, clamped to [0,1].
    Transparent where dim, opaque where bright — additive-safe compositing.
    """
    rgba = np.clip(buf, 0., 1.)
    rgba[..., 3] = np.clip(rgba[..., :3].max(axis=-1) * 1.5, 0., 1.)
    ax.imshow(rgba, extent=extent, origin='lower',
              interpolation='bilinear', zorder=float(zorder), aspect='auto')


# ── Math primitives ───────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _lerp_color(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def _fmt(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def _gauss(r2, sigma):
    """Gaussian kernel from squared distance."""
    return np.exp(-r2 / (2.0 * sigma * sigma + 1e-12))


def _moffat(r, alpha=1.0, beta=2.5):
    """Moffat PSF.  I(r) = (1 + (r/α)²)^(−β). Better stellar shape than Gaussian."""
    return (1.0 + (r / (float(alpha) + 1e-8)) ** 2) ** (-float(beta))


def _capsule_dist(X, Y, x0, y0, x1, y1):
    """
    Exact distance to a 2-D line segment (Inigo Quilez capsule SDF).
    https://iquilezles.org/articles/distfunctions2d/
    """
    ddx, ddy = x1 - x0, y1 - y0
    px, py = X - x0, Y - y0
    h = np.clip((px*ddx + py*ddy) / (ddx*ddx + ddy*ddy + 1e-12), 0., 1.)
    qx = px - ddx * h
    qy = py - ddy * h
    return np.sqrt(qx*qx + qy*qy + 1e-12).astype(np.float32)


def _fbm2d(X, Y, seed, octaves=5, lacunarity=2.0, gain=0.5, scale=3.0):
    """
    2-D fractional Brownian motion via random-phase trigonometric series.
    Returns float32 in roughly [−1, 1].
    """
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    val = np.zeros_like(X, dtype=np.float32)
    amp, freq = 0.5, 1.0 / scale
    for _ in range(octaves):
        kx = rng.uniform(-1, 1) * freq
        ky = rng.uniform(-1, 1) * freq
        ph = rng.uniform(0, 2 * math.pi)
        val += amp * np.sin(kx * X + ky * Y + ph).astype(np.float32)
        amp *= gain
        freq *= lacunarity
    return val


def _blackbody_rgb(T_kelvin):
    """Mitchell's fast blackbody → linear RGB.  T in [1 000, 40 000] K."""
    t = max(1000., min(40000., float(T_kelvin))) / 100.
    # Red
    r = 1.0 if t <= 66 else \
        float(np.clip(329.698727446 * (t - 60) ** -0.1332047592 / 255., 0., 1.))
    # Green
    g = float(np.clip((99.4708025861 * math.log(t) - 161.1195681661) / 255., 0., 1.)) \
        if t <= 66 else \
        float(np.clip(288.1221695283 * (t - 60) ** -0.0755148492 / 255., 0., 1.))
    # Blue
    b = 1.0 if t >= 66 else \
        (0.0 if t <= 19 else
         float(np.clip((138.5177312231 * math.log(t - 10) - 305.0447927307) / 255., 0., 1.)))
    return (r, g, b)


def _add_rgb(buf, field, col):
    """Additive-blend a 2-D float field into buf's RGB channels."""
    f = field.astype(np.float32)
    buf[..., 0] += f * float(col[0])
    buf[..., 1] += f * float(col[1])
    buf[..., 2] += f * float(col[2])


# ── Phenomenon renderers ──────────────────────────────────────────────────────

def _comet(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    Near-parabolic comet flyby: single open-trajectory pass across the canvas.
    True anomaly ν sweeps linearly from ν_start (entry) through 0 (perihelion)
    to ν_end (exit). The comet enters from one side, makes its closest approach
    to the orbital focus (soul element), then exits on the other side — never
    looping. Tail always points anti-solar (radially away from focus), so it
    visibly sweeps direction as the comet rounds perihelion.
    """
    if intensity < 0.05:
        return
    rng = np.random.RandomState(int(rng_seed) & 0x7FFFFFFF)
    col = _hex_to_rgb(color_hex)

    # ── Orbital elements for open-trajectory flyby ───────────────────────────
    # e in [0.75, 0.90]: very eccentric but NOT near-parabolic.
    # Near-parabolic (e > 0.95) sends the comet 40+ WU from focus at ν=±140°,
    # far outside the 26 WU canvas and invisible for most of the animation.
    # With e=0.80 and ν sweep of 200°, r at the ends ≈ 6–9 WU — always in frame.
    e   = float(np.clip(0.75 + rng.uniform(0, 0.15), 0.75, 0.90))
    q   = 1.5 + intensity * 1.0          # perihelion distance from focus (WU)
    p   = q * (1.0 + e)                  # semi-latus rectum: r = p/(1+e·cos ν)
    inc = rng.uniform(0.0, 0.45)         # inclination (Y-axis projection tilt)
    arg = rng.uniform(0, 2 * math.pi)   # orbit orientation — sets entry/exit direction

    # Perihelion falls between 40–60 % into the animation (always near centre
    # of time axis so the bright approach is fully visible).
    peri_frac = 0.40 + rng.uniform(0, 0.20)

    # ν sweep: 190°–230° total arc. With e=0.80 and ν=±100°, r ≈ 4–6 WU from
    # focus — the comet enters and exits the canvas visibly at both ends.
    nu_sweep = math.radians(190.0 + rng.uniform(0, 40.0))
    nu_start = -nu_sweep * peri_frac              # ν at frame 0  (negative = pre-perihelion)
    nu_end   =  nu_sweep * (1.0 - peri_frac)      # ν at last frame (positive = post-perihelion)

    focus_x, focus_y = cx, cy

    # ── True anomaly at this frame (linear sweep) ─────────────────────────────
    # Linear-in-ν is slightly non-Keplerian (physically the comet moves fastest
    # at perihelion), but it ensures the close approach is visible for multiple
    # frames rather than flashing through in 1–2 frames.
    t_frac = frame_idx / max(n_frames - 1, 1)
    nu     = nu_start + t_frac * (nu_end - nu_start)   # current true anomaly

    # ── Position from conic section  r = p / (1 + e·cos ν) ──────────────────
    denom = max(1.0 + e * math.cos(nu), 1e-4)
    r_orb = p / denom
    nu_rot = nu + arg
    hx = float(np.clip(focus_x + r_orb * math.cos(nu_rot), 0.3, W - 0.3))
    hy = float(np.clip(focus_y + r_orb * math.sin(nu_rot) * math.cos(inc), 0.3, H - 0.3))

    # ── Anti-solar tail: radially AWAY from the orbital focus ─────────────────
    solar_dx = hx - focus_x
    solar_dy = hy - focus_y
    solar_r  = math.sqrt(solar_dx**2 + solar_dy**2 + 1e-8)
    tx = solar_dx / solar_r
    ty = solar_dy / solar_r

    # Cometary activity: brightest at perihelion (ν=0), dims with distance.
    # activity = (q/r_orb)^0.35 = ((1+e·cos ν)/(1+e))^0.35
    activity = float(((1.0 + e * math.cos(nu)) / (1.0 + e)) ** 0.35)

    # Brightness = base floor (always off-gassing) + perihelion boost.
    # Decouples positional physics from visibility: comet is always legible,
    # but flares dramatically at closest approach.
    nuc_bright = float(np.clip(intensity * (0.55 + 0.45 * activity), 0., 1.))

    # Tail length varies more dramatically (grows 2× from aphelion to perihelion)
    # Floor ensures comet always looks comet-like, not a stellar point
    tail_len   = max((1.5 + intensity * 1.5) * (0.35 + 0.65 * activity), 0.7)

    r_w = tail_len + 1.8
    buf, extent, N = _patch(hx, hy, r_w, fig_sz, W, H, min_n=96)
    X, Y = _coords(N, extent)

    # ── Dust tail (warm colour, broad, curved by radiation pressure) ──────────
    # Radiation pressure deflects dust slightly perpendicular to the radial direction
    dust_tx = tx + ty * 0.20
    dust_ty = ty - tx * 0.20
    dn = math.sqrt(dust_tx**2 + dust_ty**2 + 1e-8)
    dust_tx /= dn; dust_ty /= dn

    t1x = hx + dust_tx * tail_len
    t1y = hy + dust_ty * tail_len
    d_dust = _capsule_dist(X, Y, hx, hy, t1x, t1y)

    proj_d = (X - hx) * dust_tx + (Y - hy) * dust_ty
    taper_d = np.clip(1.0 - proj_d / (tail_len + 1e-6), 0., 1.) ** 0.70
    # Dust tail brightness: floor at 0.55 so tail stays visible at aphelion.
    # Physically: comets always off-gas at some level; visually: asymmetry is
    # what makes it read as a comet rather than a star.
    dust_floor = 0.55 + intensity * 0.15    # 0.55–0.70 depending on intensity
    tail_bright_d = nuc_bright * 0.50 + dust_floor * 0.50  # blend, floor-dominant
    dust_r = 0.50 + intensity * 0.35        # wider dust column for visibility
    dust_bright = np.exp(-d_dust / (dust_r * 0.45)) * taper_d * tail_bright_d

    warm = (min(col[0] * 1.3 + 0.18, 1.), col[1] * 0.78, col[2] * 0.32)
    _add_rgb(buf, dust_bright, warm)

    # ── Ion tail (blue-white, narrow, strictly anti-solar = straight radial) ──
    ion_len = tail_len * 0.75
    i1x = hx + tx * ion_len
    i1y = hy + ty * ion_len
    d_ion = _capsule_dist(X, Y, hx, hy, i1x, i1y)
    proj_i = (X - hx) * tx + (Y - hy) * ty
    taper_i = np.clip(1.0 - proj_i / (ion_len + 1e-6), 0., 1.) ** 0.55
    ion_r = 0.16 + intensity * 0.12
    ion_floor = 0.45 + intensity * 0.10
    tail_bright_i = nuc_bright * 0.50 + ion_floor * 0.50
    ion_bright = np.exp(-d_ion / (ion_r * 0.40)) * taper_i * tail_bright_i
    ion_col = (col[0] * 0.42, col[1] * 0.68, min(col[2] * 1.6 + 0.38, 1.))
    _add_rgb(buf, ion_bright, ion_col)

    # ── 3-layer Gaussian coma ─────────────────────────────────────────────────
    # Sigma is FIXED (coma's physical size is set by nucleus sublimation radius,
    # not solar distance). Only brightness scales with nuc_bright.
    # Strengths are high enough so the outer (sigma=1.1) halo is perceptible at aphelion:
    #   outer alpha at centre ≈ 0.65 × nuc_bright × 1.5 ≥ 0.65 everywhere.
    ddx = X - hx; ddy = Y - hy
    r2 = ddx*ddx + ddy*ddy
    for sigma, strength in [(1.10, 0.65), (0.42, 0.85), (0.12, 1.00)]:
        g = _gauss(r2, sigma) * strength * nuc_bright
        coma_col = _lerp_color(col, (1., 1., .90), strength * 0.55)
        _add_rgb(buf, g, coma_col)

    # ── Nucleus: bright Moffat PSF (flares at perihelion) ─────────────────────
    r = np.sqrt(r2).astype(np.float32)
    nucleus = _moffat(r, alpha=0.12 + nuc_bright * 0.08, beta=2.5) * nuc_bright
    buf[..., 0] += nucleus.astype(np.float32)
    buf[..., 1] += (nucleus * 0.94).astype(np.float32)
    buf[..., 2] += (nucleus * 0.78).astype(np.float32)

    _show(ax, buf, extent, zorder=6)


def _supernova(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    3-phase supernova lifecycle driven by animation phase:
    • 0–30 %  : compact pre-flash stellar glow
    • 5–60 %  : white bloom (sin envelope, peaks at ~30 %)
    • 10–100%  : Sedov-Taylor shell growing as t^0.4 + quadrupolar fBm filaments
    """
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)
    phase = frame_idx / max(n_frames - 1, 1)  # 0 → 1 across animation

    r_w = 2.5 + intensity * 1.2          # reduced: was 4.0 + 2.0·intensity
    buf, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=96)
    X, Y = _coords(N, extent)
    ddx = X - cx; ddy = Y - cy
    R = np.sqrt(ddx*ddx + ddy*ddy + 1e-12).astype(np.float32)
    theta = np.arctan2(ddy, ddx).astype(np.float32)

    # ── Pre-flash stellar glow (0–30 %) ───────────────────────────────────────
    pre_env = float(math.sin(max(0., min(1., phase / 0.30)) * math.pi))
    pre_str = pre_env * intensity
    if pre_str > 0.005:
        pre = _moffat(R, alpha=0.10, beta=2.0) * pre_str * 1.8
        _add_rgb(buf, pre, col)

    # ── White bloom (5–60 %, peaks ~30 %) ────────────────────────────────────
    bloom_t = max(0., min(1., (phase - 0.05) / 0.55))
    bloom_env = float(math.sin(bloom_t * math.pi))
    bloom_str = bloom_env * intensity
    if bloom_str > 0.005:
        for sigma, frac in [(0.12, 1.0), (0.38, 0.65), (0.80, 0.28)]:  # reduced sigmas
            g = _gauss(R * R, sigma) * bloom_str * frac
            buf[..., 0] += g.astype(np.float32)
            buf[..., 1] += g.astype(np.float32)
            buf[..., 2] += g.astype(np.float32)

    # ── Sedov-Taylor expanding shell (10–100 %) ───────────────────────────────
    shell_t = max(0., (phase - 0.10)) / 0.90
    shell_r = 0.28 + (shell_t ** 0.40) * (1.6 + intensity * 0.9)  # reduced: was 0.40+(2.6+1.5·i)
    shell_d = 0.20 + intensity * 0.12

    dist_shell = np.abs(R - shell_r).astype(np.float32)
    # Limb brightening: sharp inner rim + soft outer fall
    limb = np.exp(-(dist_shell / (shell_d * 0.18)) ** 2)
    soft = np.exp(-dist_shell / (shell_d * 0.65))
    shell_field = (limb * 0.65 + soft * 0.35).astype(np.float32)

    # Quadrupolar angular modulation (explosion asymmetry)
    quad = (1.0 + 0.30 * np.cos(2.0 * theta)).astype(np.float32)
    shell_field *= quad

    shell_vis = float(min(1., max(0., (phase - 0.10) / 0.20)))
    shell_alpha = intensity * 0.72 * shell_vis
    sh_col = _lerp_color(col, (1., 1., 1.), 0.40)
    _add_rgb(buf, shell_field * shell_alpha, sh_col)

    # ── fBm ejecta filaments (from 25 % onward) ───────────────────────────────
    if phase > 0.25 and intensity > 0.14:
        fbm = _fbm2d(X, Y, rng_seed, octaves=4, scale=1.8)
        fbm_mask = np.clip((fbm - 0.05) * 4.0, 0., 1.).astype(np.float32)
        zone = np.exp(-dist_shell / (shell_d * 2.5))
        fil_vis = float(min(1., (phase - 0.25) / 0.22))
        filament = fbm_mask * zone * intensity * 0.52 * fil_vis
        fil_col = _lerp_color(col, (1., 0.50, 0.12), 0.55)
        _add_rgb(buf, filament, fil_col)

    _show(ax, buf, extent, zorder=5)


def _globular_cluster(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H,
                      density_mult=1.0):
    """
    King (1962) density profile: continuous glow field + rejection-sampled
    discrete stars rendered with Moffat PSF.
    """
    if intensity < 0.05:
        return
    rng = np.random.RandomState(int(rng_seed) & 0x7FFFFFFF)
    col = _hex_to_rgb(color_hex)

    r_c = 0.28 + intensity * 0.22   # King core radius
    r_t = 1.5 + intensity * 1.5     # King tidal radius
    r_w = r_t + 0.6
    buf, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=96)
    X, Y = _coords(N, extent)
    ddx = X - cx; ddy = Y - cy
    R = np.sqrt(ddx*ddx + ddy*ddy + 1e-12).astype(np.float32)

    # ── King (1962) continuous density field ──────────────────────────────────
    C_val = float(1.0 / math.sqrt(1.0 + (r_t / r_c) ** 2))
    with np.errstate(invalid='ignore'):
        term = np.maximum(0., 1.0 / np.sqrt(1.0 + (R / r_c) ** 2) - C_val)
    king = (term ** 2).astype(np.float32)
    king_max = float(king.max()) + 1e-8
    king /= king_max

    # Gentle twinkling of the integrated glow
    tw = 0.88 + 0.12 * math.sin(frame_idx * 0.28 + float(rng_seed) * 0.01)
    _add_rgb(buf, king * intensity * 0.40 * tw, col)

    # ── Discrete stars via King-profile rejection sampling ────────────────────
    n_stars = min(80, int((15 + intensity * 65) * density_mult))
    n_try = n_stars * 22
    xt = rng.uniform(-r_t, r_t, n_try).astype(np.float32)
    yt = rng.uniform(-r_t, r_t, n_try).astype(np.float32)
    Rt = np.sqrt(xt*xt + yt*yt)
    in_rt = Rt < r_t
    xt = xt[in_rt]; yt = yt[in_rt]; Rt = Rt[in_rt]
    term_t = np.maximum(0., 1.0 / np.sqrt(1.0 + (Rt / r_c) ** 2) - C_val)
    density_t = term_t ** 2
    max_d = (1.0 - C_val) ** 2 + 1e-12
    accept_u = rng.uniform(0, max_d, len(Rt))
    accepted = accept_u < density_t
    xacc = (cx + xt[accepted])[:n_stars]
    yacc = (cy + yt[accepted])[:n_stars]
    brights = rng.uniform(0.25, 1.0, len(xacc))

    ppu_val = _ppu(fig_sz, W)
    k_rad = max(2, min(6, int(0.15 * ppu_val)))
    k_sigma = max(0.5, ppu_val * 0.06)

    for sx, sy, sb in zip(xacc, yacc, brights):
        pi = int(round((sy - (cy - r_w)) / (2 * r_w) * N))
        pj = int(round((sx - (cx - r_w)) / (2 * r_w) * N))
        i0 = max(0, pi - k_rad); i1 = min(N, pi + k_rad + 1)
        j0 = max(0, pj - k_rad); j1 = min(N, pj + k_rad + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        ki = np.arange(i0, i1) - pi
        kj = np.arange(j0, j1) - pj
        KJ, KI = np.meshgrid(kj, ki)
        kr = np.sqrt(KI**2 + KJ**2 + 0.).astype(np.float32)
        psf = _moffat(kr, alpha=k_sigma, beta=2.5) * float(sb) * intensity
        hue = rng.uniform(-0.08, 0.08)
        sc = (min(col[0] + hue + 0.14, 1.), col[1] + 0.06, max(col[2] - hue + 0.16, 0.))
        buf[i0:i1, j0:j1, 0] += (psf * sc[0]).astype(np.float32)
        buf[i0:i1, j0:j1, 1] += (psf * sc[1]).astype(np.float32)
        buf[i0:i1, j0:j1, 2] += (psf * sc[2]).astype(np.float32)

    # Central condensation
    core = _moffat(R, alpha=r_c * 0.40, beta=1.8) * intensity * 0.78
    buf[..., 0] += (core * 1.00).astype(np.float32)
    buf[..., 1] += (core * 0.97).astype(np.float32)
    buf[..., 2] += (core * 0.88).astype(np.float32)

    _show(ax, buf, extent, zorder=5)


def _meteor(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H,
            freq_mult=1.0):
    """
    Tapered capsule SDF streak: ablation colour gradient (white-hot head →
    element colour → transparent tail) + luminous Moffat head glow.
    """
    if intensity < 0.10:
        return
    rng = np.random.RandomState(int(rng_seed) & 0x7FFFFFFF)
    col = _hex_to_rgb(color_hex)

    period = max(4, int(20 / max(intensity * freq_mult, 0.10)))
    local_f = frame_idx % period
    if local_f > 7:
        return   # streak visible only for 7 frames per period

    streak_phase = local_f / 7.0
    rng2 = np.random.RandomState(int(rng_seed + frame_idx // period) & 0x7FFFFFFF)
    angle = rng2.uniform(0.20, math.pi - 0.20)
    length = 2.0 + intensity * 2.5
    ox = float(np.clip(cx + rng2.uniform(-1.0, 1.0), 0.5, W - 0.5))
    oy = float(np.clip(cy + rng2.uniform(-1.0, 1.0), 0.5, H - 0.5))

    dx_dir = math.cos(angle); dy_dir = math.sin(angle)
    x0 = ox - dx_dir * length * 0.5
    y0 = oy - dy_dir * length * 0.5
    x1_full = ox + dx_dir * length * 0.5
    y1_full = oy + dy_dir * length * 0.5

    # Head position at current phase
    hx = x0 + (x1_full - x0) * streak_phase
    hy = y0 + (y1_full - y0) * streak_phase

    mcx = (x0 + hx) * 0.5
    mcy = (y0 + hy) * 0.5
    r_w = length * 0.5 + 1.2
    buf, extent, N = _patch(mcx, mcy, r_w, fig_sz, W, H, min_n=48)
    X, Y = _coords(N, extent)

    # Capsule SDF from tail start (x0,y0) to current head (hx,hy)
    d_streak = _capsule_dist(X, Y, x0, y0, hx, hy)

    # Projection onto trail axis: 0=tail, 1=head
    tl = math.sqrt((hx-x0)**2 + (hy-y0)**2 + 1e-8)
    proj = ((X - x0)*(hx - x0) + (Y - y0)*(hy - y0)) / tl
    head_frac = np.clip(proj / (tl + 1e-6), 0., 1.).astype(np.float32)
    taper = head_frac ** 0.65

    streak_r = 0.07 + intensity * 0.06
    streak_bright = np.exp(-d_streak / (streak_r * 0.50)) * taper * intensity

    # Ablation gradient: white at head → element colour at tail
    for ch_i, (cw, cc) in enumerate(zip((1., 1., 1.), col)):
        ch_val = cw * head_frac + cc * (1. - head_frac)
        buf[..., ch_i] += (streak_bright * ch_val).astype(np.float32)

    # Luminous glowing head
    dh = np.sqrt((X - hx)**2 + (Y - hy)**2 + 1e-12).astype(np.float32)
    fade = float(max(0., 1. - streak_phase ** 0.5))
    head_glow = _moffat(dh, alpha=0.10 * intensity + 0.02, beta=2.0) * intensity * (0.5 + fade * 0.5)
    buf[..., 0] += head_glow.astype(np.float32)
    buf[..., 1] += (head_glow * 0.90).astype(np.float32)
    buf[..., 2] += (head_glow * 0.62).astype(np.float32)

    _show(ax, buf, extent, zorder=6)


def _grav_ripple(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    Quadrupolar gravitational wave: expanding concentric rings with cos²(2θ)
    angular modulation and 1/r amplitude falloff from source.
    """
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)

    r_w = 3.0 + intensity * 2.0
    buf, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=128)
    X, Y = _coords(N, extent)
    ddx = X - cx; ddy = Y - cy
    R = np.sqrt(ddx*ddx + ddy*ddy + 1e-12).astype(np.float32)
    theta = np.arctan2(ddy, ddx).astype(np.float32)

    # Quadrupolar angular factor — 4 lobes, zero at 45° (GW strain pattern)
    quad = (np.cos(2.0 * theta) ** 2).astype(np.float32)

    n_rings = 4
    for i in range(n_rings):
        ring_phase = (frame_idx / max(n_frames - 1, 1) + i / n_rings) % 1.0
        ring_r = 0.35 + ring_phase * (r_w - 0.35)
        ring_w = 0.14 + intensity * 0.10

        dist_ring = np.abs(R - ring_r).astype(np.float32)
        ring_mask = np.exp(-dist_ring / ring_w)

        # 1/r amplitude falloff only — physical GW propagation in flat spacetime.
        # No additional phase-based damping (GW don't decay in vacuum).
        amp = intensity * 0.72 / (R + 0.5)
        ring_field = (ring_mask * quad * amp).astype(np.float32)
        _add_rgb(buf, ring_field, col)

    # Pulsing centre source
    pulse = float(0.45 + 0.55 * math.cos(frame_idx * 0.45))
    centre = _moffat(R, alpha=0.18, beta=2.0) * intensity * 0.55 * pulse
    _add_rgb(buf, centre, col)

    _show(ax, buf, extent, zorder=4.8)


def _satellite(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    Keplerian orbit (Newton solver for eccentric anomaly) + fading dot trail
    of past positions + solar glint cross-shaped diffraction spike.
    """
    if intensity < 0.05:
        return
    rng = np.random.RandomState(int(rng_seed) & 0x7FFFFFFF)
    col = _hex_to_rgb(color_hex)

    # Fixed orbital elements per element
    a     = 1.6 + intensity * 1.8
    e     = float(np.clip(rng.uniform(0.05, 0.30), 0., 0.9))
    inc   = rng.uniform(0.10, 0.45)
    period = n_frames * (0.55 + rng.uniform(0., 0.45))

    def kepler_pos(f_idx):
        M = (f_idx / period) * 2 * math.pi
        Ev = M
        for _ in range(6):   # Newton's method: E − e·sin(E) = M
            Ev -= (Ev - e * math.sin(Ev) - M) / (1. - e * math.cos(Ev) + 1e-12)
        r_orb = a * (1. - e * math.cos(Ev))
        nu = math.atan2(math.sqrt(max(0., 1. - e*e)) * math.sin(Ev), math.cos(Ev) - e)
        sx_ = float(np.clip(cx + r_orb * math.cos(nu), 0.3, W - 0.3))
        sy_ = float(np.clip(cy + r_orb * math.sin(nu) * math.cos(inc), 0.3, H - 0.3))
        return sx_, sy_

    sx, sy = kepler_pos(frame_idx)
    r_w = 2.2 + intensity * 1.5
    buf, extent, N = _patch(sx, sy, r_w, fig_sz, W, H, min_n=64)
    X, Y = _coords(N, extent)
    x0_e, x1_e, y0_e, y1_e = extent

    # ── Fading past-position trail ────────────────────────────────────────────
    trail_len = 12
    for t_back in range(1, trail_len + 1):
        t_idx = max(0, frame_idx - t_back * 2)
        tx_t, ty_t = kepler_pos(t_idx)
        if not (x0_e <= tx_t <= x1_e and y0_e <= ty_t <= y1_e):
            continue
        fade = (1. - t_back / trail_len) ** 1.8 * intensity * 0.35
        if fade < 0.005:
            continue
        dt = np.sqrt((X - tx_t)**2 + (Y - ty_t)**2 + 1e-12).astype(np.float32)
        _add_rgb(buf, _moffat(dt, alpha=0.055, beta=2.5) * fade, col)

    # ── Main satellite body ───────────────────────────────────────────────────
    ds = np.sqrt((X - sx)**2 + (Y - sy)**2 + 1e-12).astype(np.float32)
    body = _moffat(ds, alpha=0.04, beta=3.0) * intensity
    buf[..., 0] += body.astype(np.float32)
    buf[..., 1] += (body * 0.97).astype(np.float32)
    buf[..., 2] += (body * 0.82).astype(np.float32)

    # ── Solar glint: cross-shaped diffraction spike ────────────────────────────
    glint = float(0.45 + 0.55 * math.cos(frame_idx * 0.70 + float(rng_seed) * 0.1))
    dxs = X - sx; dys = Y - sy
    for angle_deg in (0, 90):
        rad = math.radians(angle_deg)
        along = dxs * math.cos(rad) + dys * math.sin(rad)
        perp  = np.abs(-dxs * math.sin(rad) + dys * math.cos(rad))
        spike  = (np.exp(-perp / 0.035) *
                  np.exp(-np.abs(along) / 0.30) *
                  intensity * glint * 0.55).astype(np.float32)
        buf[..., 0] += spike
        buf[..., 1] += (spike * 0.97).astype(np.float32)
        buf[..., 2] += (spike * 0.80).astype(np.float32)

    _show(ax, buf, extent, zorder=6)


def _dark_cloud(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    fBm domain-warped dark nebula: Beer-Lambert absorption opacity
    (transmission = exp(−ρκ)) + faint element-colour edge glow.
    Rendered as two stacked imshow layers (dark absorber + glow).
    """
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)

    r_w = 2.5 + intensity * 2.0
    buf_g, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=96)
    X, Y = _coords(N, extent)

    breath = float(1.0 + 0.06 * math.sin(frame_idx * 0.14 + float(rng_seed) * 0.02))

    # ── fBm domain warping (two-stage for fractal filament edges) ─────────────
    warp_scale = 2.0
    fx1 = _fbm2d(X, Y, rng_seed,      octaves=4, scale=warp_scale)
    fy1 = _fbm2d(X, Y, rng_seed + 7,  octaves=4, scale=warp_scale)
    warp_str = 0.55 * intensity
    Xw = X + fx1 * warp_str
    Yw = Y + fy1 * warp_str
    density_fbm = _fbm2d(Xw, Yw, rng_seed + 13, octaves=5, scale=warp_scale * 0.75)

    raw_density = np.clip(density_fbm * 0.5 + 0.5, 0., 1.).astype(np.float32)

    # Radial falloff: cloud confined to patch
    ddx = (X - cx) / r_w; ddy = (Y - cy) / r_w
    R_norm = np.sqrt(ddx*ddx + ddy*ddy).astype(np.float32)
    radial = np.clip(1. - R_norm * breath, 0., 1.) ** 1.8

    density = raw_density * radial * intensity

    # ── Beer-Lambert absorption layer ─────────────────────────────────────────
    kappa = 3.5
    cloud_opacity = np.clip(1. - np.exp(-density * kappa), 0., 1.).astype(np.float32)
    dark_buf = np.zeros((N, N, 4), dtype=np.float32)
    dark_buf[..., 3] = cloud_opacity
    ax.imshow(np.clip(dark_buf, 0., 1.), extent=extent, origin='lower',
              interpolation='bilinear', zorder=3.5, aspect='auto')

    # ── Edge glow from element colour ─────────────────────────────────────────
    # Glow at intermediate density (cloud boundary region)
    edge_zone = np.clip(1. - np.abs(density * 2. - 0.80), 0., 1.) * radial
    _add_rgb(buf_g, edge_zone * intensity * 0.48, col)
    _show(ax, buf_g, extent, zorder=3.6)


def _aurora_surge(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    Vertical aurora curtain: triangle-wave column noise (animated) +
    altitude-layered colour (purple/N₂ → green/O → red/O⁺) +
    Gaussian blur for soft glow.
    """
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)

    r_w = 2.5 + intensity * 2.0
    buf, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=96)
    X, Y = _coords(N, extent)

    x_n = (X - (cx - r_w)) / (2 * r_w)   # 0 = left, 1 = right
    y_n = (Y - (cy - r_w)) / (2 * r_w)   # 0 = bottom, 1 = top
    y_above = (Y - cy) / r_w             # −1 → +1 (0 = element centre)

    anim = frame_idx * 0.14

    # ── Triangle-wave curtain columns ─────────────────────────────────────────
    def tri(freq, seed_off):
        scaled = x_n * freq + anim + seed_off
        frac = scaled - np.floor(scaled)
        return (2. * np.abs(frac - 0.5)).astype(np.float32)

    curtain = (tri(2.5,  float(rng_seed) * 0.011)        * 0.50 +
               tri(5.5,  float(rng_seed) * 0.017 + 1.30) * 0.30 +
               tri(11.0, float(rng_seed) * 0.031 + 2.70) * 0.20)

    # ── Vertical profile: band above element centre ────────────────────────────
    vert = (np.exp(-((y_above - 0.30) ** 2) / 0.22) *
            np.clip(y_n * 4.0, 0., 1.)).astype(np.float32)

    base = curtain * vert  # (N, N) brightness field

    # ── Altitude colour layers ────────────────────────────────────────────────
    t_green = (np.clip(y_above * 3.0, 0., 1.) *
               np.clip((0.9 - y_above) * 3.5, 0., 1.)).astype(np.float32)
    t_red   = np.clip((y_above - 0.30) * 4.0, 0., 1.).astype(np.float32)
    t_blue  = np.clip((0.25 - y_above) * 6.0, 0., 1.).astype(np.float32)

    green_col = _lerp_color(col, (0.08, 1.00, 0.28), 0.65)
    red_col   = _lerp_color(col, (1.00, 0.08, 0.18), 0.55)
    blue_col  = _lerp_color(col, (0.25, 0.08, 0.90), 0.55)

    r_ch = (base * (green_col[0]*t_green + red_col[0]*t_red + blue_col[0]*t_blue) * intensity).astype(np.float32)
    g_ch = (base * (green_col[1]*t_green + red_col[1]*t_red + blue_col[1]*t_blue) * intensity).astype(np.float32)
    b_ch = (base * (green_col[2]*t_green + red_col[2]*t_red + blue_col[2]*t_blue) * intensity).astype(np.float32)

    # ── Gaussian blur for soft glow ───────────────────────────────────────────
    buf[..., 0] = gaussian_filter(r_ch, sigma=2.0).astype(np.float32)
    buf[..., 1] = gaussian_filter(g_ch, sigma=2.0).astype(np.float32)
    buf[..., 2] = gaussian_filter(b_ch, sigma=2.0).astype(np.float32)

    _show(ax, buf, extent, zorder=4.5)


def _variable_star(ax, cx, cy, intensity, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H):
    """
    Fourier Cepheid light curve (asymmetric: fast rise / slow decline) +
    Mitchell blackbody temperature shift (hot-blue at peak, cool-red at min) +
    Moffat PSF + 4-point diffraction spikes.
    """
    if intensity < 0.05:
        return
    col = _hex_to_rgb(color_hex)

    # ── Fourier Cepheid pulsation ─────────────────────────────────────────────
    period = max(n_frames // 4, 8)
    phi = 2 * math.pi * frame_idx / period
    # Typical Cepheid Fourier decomposition (R21≈0.27, φ21≈4.7 rad)
    A0=0.50; A1=0.28; A2=0.07; B1=0.18; B2=0.03
    L_norm = (A0 + A1*math.cos(phi) + A2*math.cos(2*phi)
              + B1*math.sin(phi) + B2*math.sin(2*phi))
    pulse = float(np.clip(L_norm, 0., 1.))
    brightness = intensity * pulse
    if brightness < 0.01:
        return

    # ── Blackbody colour shift ─────────────────────────────────────────────────
    T = 4500. + (8500. - 4500.) * pulse
    bb = _blackbody_rgb(T)
    star_col = _lerp_color(col, bb, 0.45)

    r_w = 0.8 + brightness * 1.5
    buf, extent, N = _patch(cx, cy, r_w, fig_sz, W, H, min_n=64)
    X, Y = _coords(N, extent)
    ddx = X - cx; ddy = Y - cy
    R = np.sqrt(ddx*ddx + ddy*ddy + 1e-12).astype(np.float32)

    # ── Outer aura ────────────────────────────────────────────────────────────
    aura = _moffat(R, alpha=0.30 * brightness + 0.08, beta=1.6) * brightness * 0.45
    _add_rgb(buf, aura, star_col)

    # ── Moffat PSF core ───────────────────────────────────────────────────────
    alpha_psf = 0.05 + brightness * 0.07
    psf = _moffat(R, alpha=alpha_psf, beta=2.5) * brightness
    _add_rgb(buf, psf, star_col)

    # White-hot inner core
    core = _moffat(R, alpha=alpha_psf * 0.25, beta=3.5) * brightness * 0.85
    buf[..., 0] += core.astype(np.float32)
    buf[..., 1] += core.astype(np.float32)
    buf[..., 2] += core.astype(np.float32)

    # ── 4-point diffraction spikes (2 perpendicular axes) ─────────────────────
    spike_len = 0.35 + brightness * 0.90
    spike_col = _lerp_color(star_col, (1., 1., 1.), 0.35)
    for angle_deg in (0, 90):
        rad = math.radians(angle_deg)
        along = ddx * math.cos(rad) + ddy * math.sin(rad)
        perp  = np.abs(-ddx * math.sin(rad) + ddy * math.cos(rad))
        spike = (np.exp(-perp / 0.038) *
                 np.exp(-np.abs(along) / (spike_len * 0.35)) *
                 brightness * 0.55).astype(np.float32)
        _add_rgb(buf, spike, spike_col)

    _show(ax, buf, extent, zorder=5.5)


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
                          cosmos_type, color_hex, rng_seed, fig_sz, W, H,
                          cosmos_p=None):
    """
    Render one soul element as its assigned cosmic phenomenon.

    Parameters
    ----------
    ax          : matplotlib Axes
    cx, cy      : float  world-coordinate centre of this element
    intensity   : float  0–1
    frame_idx   : int
    n_frames    : int
    cosmos_type : str    one of the 9 keys in _RENDERERS
    color_hex   : str    e.g. '#30b050'
    rng_seed    : int    deterministic per-element seed
    fig_sz      : float  figsize (same as render_frame)
    W, H        : float  world canvas size
    cosmos_p    : dict   optional per-phenomenon multipliers (from cosmos_loop)
    """
    fn = _RENDERERS.get(cosmos_type)
    if fn is None:
        return

    cp = cosmos_p or {}
    _mult_key = {
        'comet':         'comet_mult',
        'supernova':     'supernova_mult',
        'grav_ripple':   'grav_ripple_mult',
        'satellite':     'satellite_speed',
        'dark_cloud':    'dark_cloud_opacity',
        'aurora_surge':  'aurora_mult',
        'variable_star': 'variable_star_amp',
    }
    mult      = float(cp.get(_mult_key.get(cosmos_type, ''), 1.0))
    glow_base = float(cp.get('cosmos_glow_base', 1.0))

    if cosmos_type == 'globular_cluster':
        density = float(cp.get('globular_density', 1.0)) * glow_base
        _globular_cluster(ax, cx, cy, float(np.clip(intensity, 0., 1.)),
                          frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H,
                          density_mult=density)
    elif cosmos_type == 'meteor':
        freq = float(cp.get('meteor_frequency', 1.0)) * glow_base
        _meteor(ax, cx, cy, float(np.clip(intensity, 0., 1.)),
                frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H,
                freq_mult=freq)
    else:
        boosted = float(np.clip(intensity * mult * glow_base, 0., 1.))
        fn(ax, cx, cy, boosted, frame_idx, n_frames, color_hex, rng_seed, fig_sz, W, H)


def cosmos_label(ax, cx, cy, label, color_hex, cosmos_type, fig_sz, W, H):
    """Draw a small italic label near the cosmos element."""
    offset = {
        'comet': 0.9, 'supernova': 1.1, 'globular_cluster': 1.2,
        'meteor': 0.7, 'grav_ripple': 0.7, 'satellite': 0.5,
        'dark_cloud': 1.0, 'aurora_surge': 0.9, 'variable_star': 0.7,
    }
    off = offset.get(cosmos_type, 0.8)
    ty = min(cy + off, H - 0.3)
    ax.text(cx, ty, label, fontsize=9.0, color=color_hex, alpha=0.72,
            ha='center', va='bottom', fontstyle='italic', fontweight='light',
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground='#000000')])
