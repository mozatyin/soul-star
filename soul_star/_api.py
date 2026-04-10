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
