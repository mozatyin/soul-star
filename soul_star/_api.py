#!/usr/bin/env python3
"""Soul Star Map API — generate_soul_star(spec, output_dir)."""

import numpy as np
from pathlib import Path
import time

from . import _engine as E

# ── Spec contract ─────────────────────────────────────────────────
# spec = {
#   "name": str,             # character display name, e.g. "郝思嘉"
#   "title": str,            # output file prefix, e.g. "Scarlett_OHara"
#   "gif_name": str,         # GIF filename, e.g. "Scarlett_生态星图.gif"
#   "W": float,              # canvas width (default 26.0)
#   "H": float,              # canvas height (default 26.0)
#   "n_frames": int,         # default 60
#   "gif_duration_ms": int,  # default 300
#   "bg_temp_curve": list,   # [(pct, val), ...]
#   "aurora_curve": list,    # [(pct, val), ...]
#   "domains": list,         # [{cx,cy,rx,ry,ang,dk,mk,bk,seed,name}, ...]
#   "relations": list,       # [{name,domain,angle,color}, ...]
#   "patterns": list,        # [{tmpl,name_cn,cx,cy,sx,sy,soul}, ...]
#   "domain_curves": list,   # [[(pct,val),...], ...] one per domain
#   "relation_curves": list, # [[(pct,val),...], ...] one per relation
#   "pattern_curves": list,  # [[(pct,val),...], ...] one per pattern
#   "story_beats": list,     # [(pct, short_name, desc), ...]
#   "lifecycle_label": dict|None,  # {name, curve, thresholds, tier_names, color}
#   "ecology": list,         # [{name,etype,birth,death,curve,excl_r,...}, ...]
# }

MIN_ECO_GAP = 1.5
_ERNG = np.random.RandomState(1337)


def _epoch_intensities(spec, pct):
    """Build intensity dict from spec curves at given pct."""
    return {
        'domain_i':  [E.lerp(c, pct) for c in spec['domain_curves']],
        'rel_i':     [E.lerp(c, pct) for c in spec['relation_curves']],
        'pat_d':     [E.lerp(c, pct) for c in spec['pattern_curves']],
    }


def generate_soul_star(spec, output_dir):
    """
    Generate animated soul star map GIF for any character.

    Parameters
    ----------
    spec : dict   — see module docstring for full schema
    output_dir : str | Path

    Returns
    -------
    Path — path to the output GIF file
    """
    W  = float(spec.get('W', 26.0))
    H  = float(spec.get('H', 26.0))
    N  = int(spec.get('n_frames', 60))
    gif_ms = int(spec.get('gif_duration_ms', 300))
    FIG_SZ = 7.0 * (W / 20.0)   # keeps star pixel size constant across canvas sizes
    DPI = 100
    px_soul = int(FIG_SZ * DPI)
    sky_events = E.precompute_sky_events(
        N, px_soul, rng_seed=42,
        story_beats=spec.get('story_beats'),
        sky_event_beats=spec.get('sky_event_beats'),
        guarantee_all=True,
    )

    universe = {
        'name':      spec['name'],
        'title':     spec['title'],
        'domains':   spec['domains'],
        'relations': spec['relations'],
        'patterns':  spec['patterns'],
    }
    ecology_elements = spec.get('ecology', [])
    bg_temp_curve   = spec['bg_temp_curve']
    aurora_curve    = spec['aurora_curve']
    story_beats     = spec['story_beats']
    lifecycle_label = spec.get('lifecycle_label')

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("═" * 62)
    print(f"  Soul Star Map API — {spec['name']}")
    print(f"  {N} frames × 100 sentences")
    print(f"  Output: {out}/")
    print("═" * 62)

    # Pre-compute fixed star/proto positions
    fixed_pos = E.precompute_positions(universe, W, H)

    # Pre-compute ecology element positions (frame-by-frame simulation)
    ecology_positions = E.precompute_ecology_positions(
        universe, N, ecology_elements, fixed_pos, W, H, _ERNG, MIN_ECO_GAP
    )

    t0 = time.time()
    frame_paths = []

    for fi in range(N):
        pct = fi * 100.0 / max(N - 1, 1)
        intensities = _epoch_intensities(spec, pct)
        char = E.build_epoch(universe, intensities, pct, ecology_elements, ecology_positions)

        out_frame = out / f"frame_{fi:03d}_pct{round(pct):03d}.png"
        E.render_frame(
            char, fixed_pos, pct, fi, out_frame,
            FIG_SZ, DPI, ecology_positions,
            W, H, bg_temp_curve, aurora_curve,
            ecology_elements, story_beats, lifecycle_label,
            sky_events=sky_events,
        )

        eco_alive = sum(1 for el in ecology_elements if E.lerp(el['curve'], pct) > 3.0)
        beat_name = min(story_beats, key=lambda b: abs(b[0] - pct))[1]
        print(f"  [{fi:02d}/{N-1}] {pct:5.1f}% | {beat_name} | "
              f"neb={len(char['nebs'])} star={len(char['surf'])} eco={eco_alive}")
        frame_paths.append(out_frame)

    gif_path = out / spec.get('gif_name', f"{spec['title']}.gif")
    print(f"\n  合成GIF ({N}帧 × {gif_ms}ms)…")
    E.make_gif(frame_paths, gif_path, duration_ms=gif_ms)
    print(f"  → {gif_path}")
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    return gif_path


def generate_cinema_soul_star(spec, output_dir, cinema_p=None, fps=24):
    """
    Generate cinematic 1920×1080 MP4 soul star video for any character.

    Renders each frame at 1080×1080 (soul_fig_sz from cinema_p), composites
    onto a 1920×1080 cinema canvas via _cinema.composite_cinema_frame(), then
    exports to MP4 via _video.make_mp4().

    Parameters
    ----------
    spec       : dict  — same schema as generate_soul_star
    output_dir : str | Path
    cinema_p   : dict | None  — CINEMA_P overrides (None = use defaults)
    fps        : int  — video frame rate, default 24

    Returns
    -------
    Path — path to the .mp4 file
    """
    import time as _time
    from . import _cinema as C
    from . import _video as V

    cp = {**C.CINEMA_P}
    if cinema_p:
        cp.update(cinema_p)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    N       = int(spec.get('n_frames', 60))
    W       = float(spec.get('W', 26.0))
    H       = float(spec.get('H', 26.0))
    DPI     = 100
    soul_fig_sz = cp['soul_fig_sz']   # 10.8 → 1080px at DPI=100
    px_soul = int(soul_fig_sz * DPI)
    sky_events = E.precompute_sky_events(
        N, px_soul, rng_seed=42,
        story_beats=spec.get('story_beats'),
        sky_event_beats=spec.get('sky_event_beats'),
        guarantee_all=True,
    )

    universe = {
        'name':      spec['name'],
        'title':     spec['title'],
        'domains':   spec['domains'],
        'relations': spec['relations'],
        'patterns':  spec['patterns'],
    }
    ecology_elements = spec.get('ecology', [])
    bg_temp_curve    = spec['bg_temp_curve']
    aurora_curve     = spec['aurora_curve']
    story_beats      = spec['story_beats']
    lifecycle_label  = spec.get('lifecycle_label')

    print("═" * 62)
    print(f"  Soul Star Cinema — {spec['name']}")
    print(f"  {N} frames → 1920×1080 MP4 @ {fps}fps")
    print("═" * 62)

    fixed_pos = E.precompute_positions(universe, W, H)
    ecology_positions = E.precompute_ecology_positions(
        universe, N, ecology_elements, fixed_pos, W, H, _ERNG, MIN_ECO_GAP
    )

    t0 = _time.time()
    soul_dir = out / '_soul_frames'
    soul_dir.mkdir(exist_ok=True)
    cinema_frames = []

    for fi in range(N):
        pct = fi * 100.0 / max(N - 1, 1)
        intensities = _epoch_intensities(spec, pct)
        char = E.build_epoch(universe, intensities, pct, ecology_elements, ecology_positions)

        soul_path = soul_dir / f"soul_{fi:03d}.png"
        E.render_frame(
            char, fixed_pos, pct, fi, soul_path,
            soul_fig_sz, DPI, ecology_positions,
            W, H, bg_temp_curve, aurora_curve,
            ecology_elements, story_beats, lifecycle_label,
            sky_events=sky_events,
        )

        bg_temp  = E.lerp(bg_temp_curve, pct)
        aurora_v = E.lerp(aurora_curve, pct)
        cinema_img = C.composite_cinema_frame(
            soul_path, fi, N, cp, bg_temp, aurora_v
        )
        cinema_frames.append(cinema_img)

        beat_name = min(story_beats, key=lambda b: abs(b[0] - pct))[1]
        print(f"  [{fi:02d}/{N-1}] {pct:5.1f}% | {beat_name} | cinema OK")

    mp4_name = spec.get('gif_name', f"{spec['title']}.gif").replace('.gif', '.mp4')
    mp4_path = out / mp4_name
    print(f"\n  Exporting MP4 ({N} frames @ {fps}fps)…")
    V.make_mp4(cinema_frames, mp4_path, fps=fps)
    print(f"  → {mp4_path}")
    print(f"  Total: {(_time.time()-t0)/60:.1f} min")
    return mp4_path


def generate_cosmos_soul_star(spec, output_dir, cinema_p=None, cosmos_p=None, fps=12):
    """
    Cosmos View: each relation/ecology renders as its assigned cosmic phenomenon
    instead of a generic star.  Domain nebulae are unchanged.

    Requires relations to have 'cosmos_type' field, ecology elements optionally
    have 'cosmos_type' (falls back to star if absent).

    Parameters — identical to generate_cinema_soul_star plus cosmos_p.
    Returns Path to .mp4 file.
    """
    import time as _time
    from . import _cinema as C
    from . import _cosmos as COS
    from . import _video as V

    cp = {**C.CINEMA_P}
    if cinema_p:
        cp.update(cinema_p)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    N       = int(spec.get('n_frames', 60))
    W       = float(spec.get('W', 26.0))
    H       = float(spec.get('H', 26.0))
    DPI     = 100
    soul_fig_sz = cp['soul_fig_sz']
    px_soul = int(soul_fig_sz * DPI)
    sky_events = E.precompute_sky_events(
        N, px_soul, rng_seed=42,
        story_beats=spec.get('story_beats'),
        sky_event_beats=spec.get('sky_event_beats'),
        guarantee_all=True,
    )

    universe = {
        'name':      spec['name'],
        'title':     spec['title'],
        'domains':   spec['domains'],
        'relations': spec['relations'],
        'patterns':  spec['patterns'],
    }
    ecology_elements = spec.get('ecology', [])
    bg_temp_curve    = spec['bg_temp_curve']
    aurora_curve     = spec['aurora_curve']
    story_beats      = spec['story_beats']
    lifecycle_label  = spec.get('lifecycle_label')

    print("═" * 62)
    print(f"  Soul Star Cosmos View — {spec['name']}")
    print(f"  {N} frames → 1920×1080 MP4 @ {fps}fps")
    print("═" * 62)

    fixed_pos = E.precompute_positions(universe, W, H)
    ecology_positions = E.precompute_ecology_positions(
        universe, N, ecology_elements, fixed_pos, W, H, _ERNG, MIN_ECO_GAP
    )

    t0 = _time.time()
    soul_dir = out / '_cosmos_frames'
    soul_dir.mkdir(exist_ok=True)
    cinema_frames = []

    for fi in range(N):
        pct = fi * 100.0 / max(N - 1, 1)
        intensities = _epoch_intensities(spec, pct)
        char = E.build_epoch(universe, intensities, pct, ecology_elements, ecology_positions)

        soul_path = soul_dir / f"cosmos_{fi:03d}.png"
        _render_cosmos_frame(
            char, fixed_pos, pct, fi, soul_path,
            soul_fig_sz, DPI, ecology_positions,
            W, H, bg_temp_curve, aurora_curve,
            ecology_elements, story_beats, lifecycle_label,
            sky_events=sky_events,
            universe=universe,
            intensities=intensities,
            n_frames=N,
            spec=spec,
            cosmos_p=cosmos_p,
        )

        bg_temp  = E.lerp(bg_temp_curve, pct)
        aurora_v = E.lerp(aurora_curve, pct)
        cinema_img = C.composite_cinema_frame(
            soul_path, fi, N, cp, bg_temp, aurora_v
        )
        cinema_frames.append(cinema_img)

        beat_name = min(story_beats, key=lambda b: abs(b[0] - pct))[1]
        print(f"  [{fi:02d}/{N-1}] {pct:5.1f}% | {beat_name} | cosmos OK")

    mp4_name = spec.get('gif_name', f"{spec['title']}.gif").replace('.gif', '_cosmos.mp4')
    mp4_path = out / mp4_name
    print(f"\n  Exporting Cosmos MP4 ({N} frames @ {fps}fps)…")
    V.make_mp4(cinema_frames, mp4_path, fps=fps)
    print(f"  → {mp4_path}")
    print(f"  Total: {(_time.time()-t0)/60:.1f} min")
    return mp4_path


def _render_cosmos_frame(char, fixed_pos, pct, frame_idx, output_path,
                         fig_sz, dpi, ecology_positions,
                         W, H, bg_temp_curve, aurora_curve,
                         ecology_elements, story_beats, lifecycle_label,
                         sky_events=None, universe=None, intensities=None,
                         n_frames=60, spec=None, cosmos_p=None):
    """
    Cosmos frame: domain nebulae unchanged, relations/ecology rendered as
    their typed cosmic phenomena instead of generic stars.
    """
    import math as _math
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as _np
    from . import _cosmos as COS

    p = E.P
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz), dpi=dpi)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)

    # ── Sky atmosphere ───────────────────────────────────────────────
    aurora_surge_mult = 1.0
    if sky_events:
        for evt in sky_events:
            if evt['type'] == 'aurora_surge' and evt['start_frame'] <= frame_idx <= evt['end_frame']:
                dur = max(evt['end_frame'] - evt['start_frame'] + 1, 1)
                phase = (frame_idx - evt['start_frame']) / max(dur - 1, 1)
                aurora_surge_mult = 1.0 + (evt['params']['peak_mult'] - 1.0) * _math.sin(_math.pi * phase)
                break
    E.render_sky_atmosphere(ax, pct, fig_sz, dpi, W, H, bg_temp_curve, aurora_curve, aurora_surge_mult)

    # ── Starfield background ─────────────────────────────────────────
    E.render_starfield_bg(ax, W, H, int(fig_sz * dpi), frame_idx, sky_events)

    # ── Vignette ─────────────────────────────────────────────────────
    px = int(fig_sz * dpi)
    yy, xx = _np.mgrid[0:px, 0:px]
    r2 = (xx / px * 2 - 1) ** 2 + (yy / px * 2 - 1) ** 2
    vig = _np.zeros((px, px, 4), dtype=_np.float32)
    vig[:, :, 3] = _np.clip((r2 - 0.15) * 0.60, 0, 0.52).astype(_np.float32)
    ax.imshow(vig, extent=[0, W, 0, H], origin='lower',
              interpolation='bilinear', zorder=2.8, aspect='auto')

    # ── Domain nebulae (unchanged from map view) ─────────────────────
    if char['nebs']:
        p_neb = p
        if sky_events:
            for evt in sky_events:
                if evt['type'] == 'nebula_breath':
                    dur = max(evt['end_frame'] - evt['start_frame'] + 1, 1)
                    nb_phase = frame_idx / max(dur - 1, 1)
                    breath = 1.0 + evt['params']['amplitude'] * _math.sin(_math.pi * nb_phase * 0.5)
                    p_neb = dict(p); p_neb['neb_amax'] = min(1.0, p['neb_amax'] * breath)
                    break
        neb = E.build_fractal_nebs(p_neb, px, char['nebs'], W, H)
        ax.imshow(neb, extent=[0, W, 0, H], origin='lower',
                  interpolation='bilinear', zorder=3, aspect='auto')
        for nd in char['nebs']:
            cx_d, cy_d, mk_col, lbl = nd[0], nd[1], nd[6], nd[10]
            ax.text(cx_d, cy_d, lbl, fontsize=11.0, color='#e8f4ff', alpha=0.68,
                    ha='center', va='center', fontstyle='italic', fontweight='light',
                    zorder=3.5,
                    path_effects=[pe.withStroke(linewidth=5.0, foreground=mk_col)])

    # ── Constellation patterns + soul stars (faint background layer) ──────────
    if universe and universe.get('patterns'):
        from soul_star._engine import TMPL
        for const in universe['patterns']:
            tmpl_name = const.get('tmpl')
            if tmpl_name not in TMPL:
                continue
            tmpl = TMPL[tmpl_name]
            cx_c, cy_c = const['cx'], const['cy']
            csx, csy = const['sx'], const['sy']
            soul_annots = const.get('soul', {})
            coords = {name: (cx_c + (nx - 0.5) * csx, cy_c + (ny - 0.5) * csy)
                      for name, ((nx, ny), *_) in tmpl['stars'].items()}
            # Connecting lines — very faint silver
            for sa, sb in tmpl.get('lines', []):
                if sa in coords and sb in coords:
                    x1, y1 = coords[sa]; x2, y2 = coords[sb]
                    ax.plot([x1, x2], [y1, y2], '-', color='#8090c0', alpha=0.15,
                            lw=0.7, solid_capstyle='round', zorder=3.6)
            # Stars
            for name, ((nx, ny), mag, spec_col) in tmpl['stars'].items():
                sx_c, sy_c = coords[name]
                is_soul = name in soul_annots
                if is_soul:
                    lbl_txt, soul_col = soul_annots[name]
                    E.draw_star_small(ax, sx_c, sy_c, 0.14, soul_col, 0.65,
                                      fig_sz, z=3.8, W=W)
                    ax.text(sx_c, sy_c + 0.28, lbl_txt, fontsize=7.5,
                            color=soul_col, alpha=0.55, ha='center', va='bottom',
                            fontstyle='italic', zorder=3.9,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground='#000000')])
                else:
                    E.draw_star_small(ax, sx_c, sy_c, 0.08, spec_col, 0.38,
                                      fig_sz, z=3.6, W=W)
            # Constellation name label
            all_x = [v[0] for v in coords.values()]
            all_y = [v[1] for v in coords.values()]
            lbl_cx = sum(all_x) / len(all_x); lbl_cy = min(all_y)
            ax.text(lbl_cx, lbl_cy - 0.4, const['name_cn'], fontsize=8.5,
                    color='#4060a0', alpha=0.50, ha='center', va='top',
                    fontweight='light', zorder=3.7,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground='#000000')])

    # ── COSMOS: relations as typed phenomena ─────────────────────────
    relations = universe['relations'] if universe else []
    rel_curves = spec.get('relation_curves', []) if spec else []
    for i, rel in enumerate(relations):
        cosmos_type = rel.get('cosmos_type')
        if not cosmos_type:
            continue   # no cosmos type → skip (won't render in cosmos view)
        pos = fixed_pos.get(rel['name'])
        if pos is None:
            continue
        cx_r, cy_r = pos
        curve = rel_curves[i] if i < len(rel_curves) else [(0, 0), (100, 0)]
        raw_intensity = E.lerp(curve, pct)
        intensity = float(_np.clip(raw_intensity / 10.0, 0, 1))
        seed = abs(hash(rel['name'])) % 99999
        COS.render_cosmos_element(
            ax, cx_r, cy_r, intensity, frame_idx, n_frames,
            cosmos_type, rel['color'], seed, fig_sz, W, H,
            cosmos_p=cosmos_p,
        )
        if intensity > 0.1:
            COS.cosmos_label(ax, cx_r, cy_r, rel['name'],
                             rel['color'], cosmos_type, fig_sz, W, H)

    # ── COSMOS: ecology elements as typed phenomena ──────────────────
    eco_curves_map = {el['name']: el['curve'] for el in ecology_elements}
    for el in ecology_elements:
        cosmos_type = el.get('cosmos_type')
        if not cosmos_type:
            continue
        birth = el.get('birth', 0); death = el.get('death', 101)
        if not (birth <= pct < death):
            continue
        pos = ecology_positions.get(el['name'])
        if pos is None:
            continue
        cx_e, cy_e = pos
        raw_intensity = E.lerp(el['curve'], pct)
        intensity = float(_np.clip(raw_intensity / 10.0, 0, 1))
        seed = abs(hash(el['name'])) % 99999
        COS.render_cosmos_element(
            ax, cx_e, cy_e, intensity, frame_idx, n_frames,
            cosmos_type, el.get('color', el.get('bk', '#a0b0ff')),
            seed, fig_sz, W, H,
            cosmos_p=cosmos_p,
        )
        if intensity > 0.1:
            COS.cosmos_label(ax, cx_e, cy_e, el['name'],
                             el.get('color', '#a0b0ff'), cosmos_type, fig_sz, W, H)

    # ── Soul node (Harry) — kept as a bright central star ───────────
    soul_r = p['sf_outer_r']
    for lbl, neb_idx, angle_deg, mag, col in char['surf']:
        # In cosmos view, only draw the soul node (Harry = centre isSoulNode=True)
        sx_c, sy_c = fixed_pos[lbl]
        dist = _math.sqrt((sx_c - W / 2) ** 2 + (sy_c - H / 2) ** 2)
        if dist < 1.0:  # soul node is near canvas centre
            sc = _np.clip((mag - 6.0) / 3.5, 0.0, 1.0); sc_p = sc ** 1.8
            r = soul_r * (0.18 + 0.82 * sc_p)
            E.draw_star_full(ax, sx_c, sy_c, r, col, p['sf_alpha'], fig_sz, z=7, W=W)

    # ── Story beat text + progress bar ──────────────────────────────
    best = min(story_beats, key=lambda b: abs(b[0] - pct))
    beat_name, beat_desc = best[1], best[2]
    ax.text(W / 2, 0.92, f"{char['name']}  宇宙观",
            fontsize=11.5, color='#aabbd0', alpha=0.65,
            ha='center', va='bottom', fontstyle='italic', zorder=11,
            path_effects=[pe.withStroke(linewidth=3.0, foreground='#000000')])
    ax.text(W / 2, 0.40,
            f"【{beat_name}】{beat_desc}",
            fontsize=7.5, color='#7888a0', alpha=0.60,
            ha='center', va='bottom', zorder=11, fontweight='light',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#000000')])

    bar_w = W * 0.70; bar_x = (W - bar_w) / 2; bar_y = 0.12
    ax.plot([bar_x, bar_x + bar_w], [bar_y, bar_y], '-',
            color='#303848', alpha=0.60, lw=1.5, zorder=11)
    ax.plot([bar_x, bar_x + bar_w * (pct / 100)], [bar_y, bar_y], '-',
            color='#6080b8', alpha=0.70, lw=1.5, zorder=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, facecolor='black', edgecolor='none')
    plt.close(fig)
