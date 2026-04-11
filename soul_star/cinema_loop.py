#!/usr/bin/env python3
"""
soul_star.cinema_loop — Cinema iteration loop with AI director panel.

CLI script: loads a character spec, runs N iteration rounds of:
  generate cinema video → pick 3 sample frames → director panel review
  → merge feedback → update CINEMA_P + ENGINE_P clone → repeat.

After the loop, bakes winning parameters back into _engine.py P dict and
does a final 60-frame render.

Usage:
    python3 soul_star/cinema_loop.py <character_slug> [--iters N] [--target SCORE] [--key API_KEY]

Examples:
    python3 soul_star/cinema_loop.py little_prince --iters 3 --target 8.0
    python3 soul_star/cinema_loop.py anna --iters 5 --target 7.5 --key sk-or-v1-...
"""

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

# Ensure soul-star root is on sys.path (needed when running as python3 soul_star/cinema_loop.py)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Spec loader ────────────────────────────────────────────────────────────

def load_spec(character_slug: str) -> dict:
    """
    Import soul_star_{character_slug} module and return its SPEC dict.

    Parameters
    ----------
    character_slug : str  e.g. 'little_prince', 'anna', 'forrest'

    Returns
    -------
    dict  the SPEC dict from the character module
    """
    # Ensure the soul-star root is on sys.path so the module is importable
    root = Path.home() / 'soul-star'
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    module_name = f'soul_star_{character_slug}'
    mod = importlib.import_module(module_name)
    if not hasattr(mod, 'SPEC'):
        raise AttributeError(f"Module {module_name} has no SPEC attribute")
    return mod.SPEC


# ── Sample frame picker ────────────────────────────────────────────────────

def pick_sample_frames(soul_dir: Path, n_frames: int) -> List[Path]:
    """
    Return up to 3 sample frame paths from soul_dir: first, middle, last.

    Parameters
    ----------
    soul_dir  : Path   directory containing PNG soul frames
    n_frames  : int    number of frames that were generated (used for index math)

    Returns
    -------
    list[Path]  1-3 paths that actually exist
    """
    soul_dir = Path(soul_dir)
    if not soul_dir.exists():
        return []

    # Collect all PNG files sorted by name
    all_frames = sorted(soul_dir.glob('*.png'))
    if not all_frames:
        return []

    n = len(all_frames)
    # Pick indices: first, middle, last (deduplicate)
    indices = sorted({0, n // 2, n - 1})
    return [all_frames[i] for i in indices if 0 <= i < n]


# ── Engine param bake-back ─────────────────────────────────────────────────

def apply_engine_params_to_file(engine_p: dict) -> None:
    """
    Bake winning engine parameters back into soul_star/_engine.py P dict.

    Only updates params that changed by >5% from the original P values.
    Preserves file formatting and type precision.

    Parameters
    ----------
    engine_p : dict  current (possibly updated) engine P values
    """
    engine_file = Path.home() / 'soul-star' / 'soul_star' / '_engine.py'
    content = engine_file.read_text()

    # Import original P to compare
    from soul_star._engine import P as ORIG_P  # noqa: PLC0415

    changed = {}
    for key, new_val in engine_p.items():
        if key not in ORIG_P:
            continue
        orig_val = ORIG_P[key]
        if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
            if abs(new_val - orig_val) / (abs(orig_val) + 1e-9) > 0.05:  # >5% change
                changed[key] = (orig_val, new_val)

    if not changed:
        print("  Engine params unchanged (all within 5% of original)")
        return

    for key, (orig, new) in changed.items():
        # Read the literal value string from file (preserves trailing zeros like 0.260)
        scan = re.search(rf"'{re.escape(key)}'\s*:\s*([0-9.eE+\-]+)", content)
        if not scan:
            print(f"  _engine.py: could not find '{key}' in file, skipping")
            continue
        file_literal = scan.group(1)
        pattern = rf"('{re.escape(key)}'\s*:\s*){re.escape(file_literal)}"
        if isinstance(orig, int):
            new_str = str(int(round(new)))
        else:
            decimals = len(file_literal.split('.')[-1]) if '.' in file_literal else 3
            new_str = f"{new:.{decimals}f}"
        content = re.sub(pattern, rf"\g<1>{new_str}", content)
        print(f"  _engine.py P['{key}']: {file_literal} → {new_str}")

    engine_file.write_text(content)
    print(f"  _engine.py updated ({len(changed)} params)")


# ── Main iteration loop ────────────────────────────────────────────────────

def run_cinema_loop(
    character_slug: str,
    max_iters: int = 20,
    target_score: float = 9.5,
    api_key: str = '',
    output_root: str = '',
    min_improvement: float = 0.2,
    plateau_rounds: int = 2,
) -> Path:
    """
    Run the cinema iteration loop for a character.

    Each iteration:
      1. Renders 40-frame cinema video (120 for final)
      2. Picks 3 sample frames from _soul_frames/
      3. Calls director panel_review()
      4. Prints review summary
      5. Saves iteration log JSON
      6. Stops if: target_score reached, OR score plateau detected, OR max_iters reached
      7. Else: merges feedback → updates cinema_p + engine_p

    Plateau detection: if score improvement < min_improvement for plateau_rounds
    consecutive rounds, the loop stops automatically.

    After loop:
      - Bakes winning engine_p into _engine.py
      - Does final 120-frame render with best cinema_p

    Parameters
    ----------
    character_slug   : str   e.g. 'little_prince'
    max_iters        : int   hard cap on iterations (default 20)
    target_score     : float stop early if panel avg >= this score (default 9.5)
    api_key          : str   Anthropic API key (or read from ANTHROPIC_API_KEY env)
    output_root      : str   base output directory (default ~/Desktop/CinemaLoop)
    min_improvement  : float minimum score gain to count as progress (default 0.2)
    plateau_rounds   : int   stop after this many consecutive low-improvement rounds (default 2)

    Returns
    -------
    Path  path to the final MP4 file
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    if api_key:
        os.environ['ANTHROPIC_API_KEY'] = api_key
    elif not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError(
            "No API key provided. Pass --key or set ANTHROPIC_API_KEY env var."
        )

    # Imports here to avoid circular issues and to allow smoke tests without GPU
    from soul_star._api import generate_cinema_soul_star
    from soul_star._cinema import CINEMA_P
    from soul_star._directors import (
        merge_feedback,
        panel_review,
        print_review_summary,
    )
    from soul_star._engine import P as ENGINE_P_ORIG

    spec = load_spec(character_slug)

    if not output_root:
        output_root = str(Path.home() / 'Desktop' / 'CinemaLoop')
    base_dir = Path(output_root) / character_slug
    base_dir.mkdir(parents=True, exist_ok=True)

    # Working copies of parameters
    cinema_p = dict(CINEMA_P)
    engine_p = dict(ENGINE_P_ORIG)

    iteration_logs = []
    best_cinema_p = dict(cinema_p)
    best_score = -1.0
    prev_score = -1.0
    plateau_count = 0
    final_mp4 = None

    print(f"\n{'='*72}")
    print(f"  CINEMA LOOP — {spec['name']} ({character_slug})")
    print(f"  max_iters={max_iters}, target_score={target_score}")
    print(f"  Output root: {base_dir}")
    print(f"{'='*72}\n")

    for iteration in range(1, max_iters + 1):
        n_frames = 40   # more frames so directors see full animation arc at 12fps

        print(f"\n--- Iteration {iteration}/{max_iters} ({n_frames} frames @ 12fps) ---")

        # ── 1. Render cinema video ─────────────────────────────────────────
        iter_dir = base_dir / f'iter_{iteration:02d}'
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Override n_frames in spec for this render
        render_spec = dict(spec)
        render_spec['n_frames'] = n_frames

        t0 = time.time()
        mp4_path = generate_cinema_soul_star(
            render_spec,
            iter_dir,
            cinema_p=cinema_p,
            fps=12,
        )
        elapsed = time.time() - t0
        print(f"  Rendered in {elapsed:.1f}s -> {mp4_path}")

        # ── 2. Pick sample frames ──────────────────────────────────────────
        soul_dir = iter_dir / '_soul_frames'
        sample_frames = pick_sample_frames(soul_dir, n_frames)
        if not sample_frames:
            print(f"  WARNING: No soul frames found in {soul_dir}, skipping review.")
            continue

        sample_frame_strs = [str(f) for f in sample_frames]
        print(f"  Sample frames: {[f.name for f in sample_frames]}")

        # ── 3. Director panel review ───────────────────────────────────────
        print(f"\n  Running director panel review (5 directors)...")
        feedbacks = panel_review(
            spec=render_spec,
            cinema_p=cinema_p,
            engine_p=engine_p,
            sample_frames=sample_frame_strs,
            iteration=iteration,
            character_desc=spec.get('title', character_slug),
        )

        # ── 4. Print review summary ────────────────────────────────────────
        avg_score = print_review_summary(feedbacks, iteration)

        if avg_score > best_score:
            best_score = avg_score
            best_cinema_p = dict(cinema_p)
            final_mp4 = mp4_path

        # ── 5. Save iteration log JSON ─────────────────────────────────────
        log_entry = {
            'iteration': iteration,
            'avg_score': avg_score,
            'n_frames': n_frames,
            'mp4_path': str(mp4_path),
            'cinema_p_snapshot': dict(cinema_p),
            'engine_p_snapshot': dict(engine_p),
            'feedbacks': [
                {
                    'director': fb.director,
                    'score': fb.score,
                    'critique': fb.critique,
                    'cinema_adjustments': fb.cinema_adjustments,
                    'engine_adjustments': fb.engine_adjustments,
                    'key_issues': fb.key_issues,
                }
                for fb in feedbacks
            ],
        }
        iteration_logs.append(log_entry)

        log_path = base_dir / 'cinema_loop_log.json'
        log_path.write_text(json.dumps(iteration_logs, indent=2, ensure_ascii=False))
        print(f"  Log saved: {log_path}")

        # ── 6. Check stopping conditions ──────────────────────────────────
        if avg_score >= target_score:
            print(f"\n  Target score {target_score} reached ({avg_score:.1f}). Stopping.")
            break

        # Plateau detection: track consecutive rounds with small improvement
        if prev_score >= 0:
            improvement = avg_score - prev_score
            if improvement < min_improvement:
                plateau_count += 1
                print(f"  Score improvement {improvement:+.1f} < {min_improvement} "
                      f"(plateau round {plateau_count}/{plateau_rounds})")
            else:
                plateau_count = 0
                print(f"  Score improvement {improvement:+.1f} — continuing.")

        prev_score = avg_score

        if plateau_count >= plateau_rounds:
            print(f"\n  Plateau detected ({plateau_rounds} consecutive low-improvement rounds). Stopping.")
            break

        # ── 7. Merge feedback for next iteration ───────────────────────────
        if iteration < max_iters:
            cinema_p, engine_p = merge_feedback(feedbacks, cinema_p, engine_p)
            print(f"  Parameters updated for iteration {iteration + 1}.")

    # ── After loop: bake engine params ────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Baking winning engine params into _engine.py...")
    apply_engine_params_to_file(engine_p)

    # ── Final 120-frame render @ 12fps = 10 seconds — cinematic slow reveal ──
    print("\n  Final 120-frame render @ 12fps (10s cinematic) with best cinema_p...")
    final_dir = base_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)

    final_spec = dict(spec)
    final_spec['n_frames'] = 120

    t0 = time.time()
    final_mp4 = generate_cinema_soul_star(
        final_spec,
        final_dir,
        cinema_p=best_cinema_p,
        fps=12,
    )
    elapsed = time.time() - t0
    print(f"  Final render done in {elapsed:.1f}s")
    print(f"  Final MP4: {final_mp4}")
    print(f"\n{'='*72}")
    print(f"  CINEMA LOOP COMPLETE")
    print(f"  Best score: {best_score:.1f}/10")
    print(f"  Output: {final_mp4}")
    print(f"{'='*72}\n")

    return final_mp4


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for cinema_loop."""
    parser = argparse.ArgumentParser(
        description='Soul Star Cinema Iteration Loop — AI director panel feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'character_slug',
        help="Character slug matching soul_star_{slug}.py (e.g. 'little_prince')",
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=20,
        metavar='N',
        help='Hard cap on iterations (default: 20)',
    )
    parser.add_argument(
        '--target',
        type=float,
        default=9.5,
        metavar='SCORE',
        help='Stop early when panel average reaches this score (default: 9.5)',
    )
    parser.add_argument(
        '--key',
        type=str,
        default='',
        metavar='API_KEY',
        help='Anthropic API key (default: read from ANTHROPIC_API_KEY env var)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        metavar='DIR',
        help='Base output directory (default: ~/Desktop/CinemaLoop)',
    )
    parser.add_argument(
        '--min-improvement',
        type=float,
        default=0.2,
        metavar='DELTA',
        help='Minimum score gain per round to count as progress (default: 0.2)',
    )
    parser.add_argument(
        '--plateau-rounds',
        type=int,
        default=2,
        metavar='N',
        help='Stop after N consecutive low-improvement rounds (default: 2)',
    )

    args = parser.parse_args()

    final_mp4 = run_cinema_loop(
        character_slug=args.character_slug,
        max_iters=args.iters,
        target_score=args.target,
        api_key=args.key,
        output_root=args.output,
        min_improvement=args.min_improvement,
        plateau_rounds=args.plateau_rounds,
    )
    print(f"Done. Final output: {final_mp4}")


if __name__ == '__main__':
    main()
