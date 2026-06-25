#!/usr/bin/env python3
"""
soul_star.cosmos_loop — Cosmos View iteration loop with AI director panel.

Each iteration:
  1. Renders N-frame cosmos video (relations/ecology as typed phenomena)
  2. Picks 3 sample frames
  3. Director panel review (cosmos-aware prompts)
  4. Merges feedback → updates CINEMA_P + COSMOS_P
  5. Repeats until target score or plateau

Usage:
    python3 soul_star/cosmos_loop.py harry_potter --iters 10 --target 9.0
"""

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Default Cosmos tuning params ──────────────────────────────────────────────

COSMOS_P = {
    # Per-phenomenon intensity multipliers (1.0 = as coded)
    'comet_mult':            1.0,   # Voldemort comet brightness/trail
    'supernova_mult':        1.0,   # Dumbledore supernova peak brightness
    'globular_density':      1.0,   # Hermione cluster star count multiplier
    'meteor_frequency':      1.0,   # Ron meteor period (higher = more frequent)
    'grav_ripple_mult':      1.0,   # Lily ring brightness
    'satellite_speed':       1.0,   # Hedwig orbit speed multiplier
    'dark_cloud_opacity':    1.0,   # Horcrux cloud opacity
    'aurora_mult':           1.0,   # Dobby/Phoenix aurora brightness
    'variable_star_amp':     1.0,   # Deathly Hallows pulse amplitude
    # Global cosmos mix
    'cosmos_glow_base':      1.0,   # base brightness multiplier for all phenomena
    'label_alpha':           0.72,  # cosmos element label opacity
}


# ── Director panel for cosmos view ────────────────────────────────────────────

def _build_cosmos_prompt(spec, cinema_p, cosmos_p, sample_frames, iteration):
    """Build a director-panel prompt specific to cosmos view."""
    name = spec.get('name', 'Character')
    title = spec.get('title', 'Soul Star')

    cosmos_assignments = """
Cosmos View — each soul element performs as its natural phenomenon:
  • 赫敏    → Globular Cluster (dense star swarm, intellect & loyalty)
  • 罗恩    → Meteor           (sporadic bright streak, wavering but returns)
  • 邓布利多 → Supernova        (brilliant flash at death, then expanding ring)
  • 伏地魔  → Comet            (dark ominous tail sweeping the sky)
  • 莉莉的爱→ Gravitational Ripple (invisible force bending everything)
  • 海德薇  → Satellite        (faithful orbit until gone)
  • 魂器    → Dark Cloud       (creeping darkness eating the light)
  • 多比    → Aurora Surge     (colorful burst, short-lived freedom)
  • 凤凰社  → Aurora Surge     (collective magical light)
  • 死亡圣器→ Variable Star    (mysterious pulse, appears in final act)
Domain nebulae (orphan pain, magic, friendship, dark, love) are unchanged."""

    prompt = f"""You are one of four master directors reviewing the COSMOS VIEW of a soul star map.

CHARACTER: {name} ({title})
ITERATION: {iteration}

COSMOS VIEW CONCEPT:
{cosmos_assignments}

This is NOT a map. It is a LIVING UNIVERSE where each character and force becomes
the cosmic phenomenon that best matches its nature. The goal is a scene that feels
like a unique, breathing cosmos — not predictable stars, but a dynamic sky where
each phenomenon tells its own story simultaneously.

DUAL STANDARD:
A. CINEMATIC QUALITY — Does this look like a $200M space opera opening?
B. NARRATIVE TRUTH — Does each phenomenon feel exactly right for what it represents?
   Voldemort's comet should feel ominous. Hermione's cluster should feel organized.
   Lily's ripple should feel everywhere yet invisible.

CURRENT PARAMETERS:
cinema_p: bgDensity={cinema_p.get('bgDensity',1.7):.2f}, nebulaGamma={cinema_p.get('nebulaGamma',0.49):.2f},
          colorTemp={cinema_p.get('colorTemp',-0.91):.2f}, vigStrength={cinema_p.get('vignette_strength',0.84):.2f}
cosmos_p: comet×{cosmos_p.get('comet_mult',1.0):.2f}, supernova×{cosmos_p.get('supernova_mult',1.0):.2f},
          globular_density×{cosmos_p.get('globular_density',1.0):.2f}, aurora×{cosmos_p.get('aurora_mult',1.0):.2f},
          dark_cloud_opacity×{cosmos_p.get('dark_cloud_opacity',1.0):.2f}, grav_ripple×{cosmos_p.get('grav_ripple_mult',1.0):.2f}

EVALUATE (score each 1–10):
1. VISUAL DRAMA — Is the overall cosmos striking and unique?
2. PHENOMENON CLARITY — Can you identify each phenomenon? Are they distinct enough?
3. NARRATIVE RESONANCE — Does each phenomenon feel TRUE to its character?
4. BALANCE — Are some phenomena overpowering others? Does the whole coexist?
5. CINEMATIC DEPTH — Layering, contrast, color harmony across the full frame?

RESPOND IN STRICT JSON:
{{
  "score": <float 1-10>,
  "critique": "<2-3 sentences overall>",
  "phenomenon_notes": {{
    "comet": "<one sentence on Voldemort comet>",
    "supernova": "<one sentence on Dumbledore>",
    "globular_cluster": "<one sentence on Hermione>",
    "grav_ripple": "<one sentence on Lily>"
  }},
  "cinema_adjustments": {{
    "bgDensity": <float or null>,
    "nebulaGamma": <float or null>,
    "colorTemp": <float or null>,
    "vignette_strength": <float or null>
  }},
  "cosmos_adjustments": {{
    "comet_mult": <float or null>,
    "supernova_mult": <float or null>,
    "globular_density": <float or null>,
    "meteor_frequency": <float or null>,
    "grav_ripple_mult": <float or null>,
    "dark_cloud_opacity": <float or null>,
    "aurora_mult": <float or null>,
    "variable_star_amp": <float or null>,
    "cosmos_glow_base": <float or null>
  }},
  "key_issues": ["<issue 1>", "<issue 2>"]
}}"""
    return prompt


def _cosmos_panel_review(spec, cinema_p, cosmos_p, sample_frames, iteration):
    """Call 4 directors and return list of feedback dicts."""
    import anthropic
    import base64

    directors = [
        ("George Lucas",  "You believe a cosmos should feel like a living myth. "
                          "Phenomena must be archetypal — Voldemort's comet is THE dark force, "
                          "not just a comet. Weight: narrative truth over beauty."),
        ("Ridley Scott",  "You demand visual magnificence. Every phenomenon must be "
                          "photorealistic-level detail in its rendering. Dark clouds must feel "
                          "ominous, auroras must be breathtaking. Weight: cinematic quality first."),
        ("Christopher Nolan", "You care about simultaneity — 10 phenomena happening at once "
                              "must each be legible. You hate visual noise masking meaning. "
                              "Weight: clarity and balance."),
        ("Steve Jobs",    "You represent the user. This is a phone screen. Does the cosmos "
                          "feel personal and magical? Would a user stop and stare? "
                          "Weight: emotional resonance + phone-scale readability."),
    ]

    base_prompt = _build_cosmos_prompt(spec, cinema_p, cosmos_p, sample_frames, iteration)
    client = anthropic.Anthropic(
        api_key=os.environ.get('ANTHROPIC_API_KEY', ''),
        base_url='https://openrouter.ai/api',
    )

    feedbacks = []
    for director_name, director_bio in directors:
        # Build image content
        content = []
        for frame_path in sample_frames[:3]:
            with open(frame_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64}
            })

        content.append({
            'type': 'text',
            'text': f"DIRECTOR: {director_name}\nBIO: {director_bio}\n\n{base_prompt}"
        })

        try:
            resp = client.messages.create(
                model='anthropic/claude-opus-4-6',
                max_tokens=1200,
                messages=[{'role': 'user', 'content': content}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception as e:
            print(f"    [{director_name}] API error: {e}")
            data = {}

        fb = {
            'director': director_name,
            'score': float(data.get('score', 5.0)),
            'critique': data.get('critique', ''),
            'phenomenon_notes': data.get('phenomenon_notes', {}),
            'cinema_adjustments': data.get('cinema_adjustments', {}),
            'cosmos_adjustments': data.get('cosmos_adjustments', {}),
            'key_issues': data.get('key_issues', []),
        }
        feedbacks.append(fb)
        print(f"    {director_name}: {fb['score']:.1f}/10")

    return feedbacks


def _merge_cosmos_feedback(feedbacks, cinema_p, cosmos_p):
    """Weighted merge of director feedback into updated cinema_p + cosmos_p."""
    weights = {
        'George Lucas': 1.3,
        'Ridley Scott': 1.1,
        'Christopher Nolan': 1.0,
        'Steve Jobs': 1.4,
    }
    total_w = sum(weights.get(fb['director'], 1.0) for fb in feedbacks)

    # ── Cinema params ──────────────────────────────────────────────
    cinema_keys = ['bgDensity', 'nebulaGamma', 'colorTemp', 'vignette_strength']
    new_cinema = dict(cinema_p)
    for key in cinema_keys:
        suggestions = []
        ws = []
        for fb in feedbacks:
            adj = fb.get('cinema_adjustments', {}) or {}
            v = adj.get(key)
            if v is not None:
                try:
                    suggestions.append(float(v))
                    ws.append(weights.get(fb['director'], 1.0))
                except (ValueError, TypeError):
                    pass
        if suggestions:
            wsum = sum(ws)
            new_val = sum(s * w for s, w in zip(suggestions, ws)) / wsum
            # Blend 35% old + 65% new
            new_cinema[key] = 0.35 * cinema_p.get(key, new_val) + 0.65 * new_val

    # ── Cosmos params ──────────────────────────────────────────────
    cosmos_keys = list(COSMOS_P.keys())
    new_cosmos = dict(cosmos_p)
    for key in cosmos_keys:
        suggestions = []
        ws = []
        for fb in feedbacks:
            adj = fb.get('cosmos_adjustments', {}) or {}
            v = adj.get(key)
            if v is not None:
                try:
                    suggestions.append(float(v))
                    ws.append(weights.get(fb['director'], 1.0))
                except (ValueError, TypeError):
                    pass
        if suggestions:
            wsum = sum(ws)
            new_val = sum(s * w for s, w in zip(suggestions, ws)) / wsum
            new_cosmos[key] = 0.35 * cosmos_p.get(key, new_val) + 0.65 * new_val

    # Clamp values to reasonable ranges
    new_cinema['bgDensity']        = max(0.3, min(4.5, new_cinema.get('bgDensity', 1.7)))
    new_cinema['nebulaGamma']      = max(0.1, min(2.0, new_cinema.get('nebulaGamma', 0.49)))
    new_cinema['colorTemp']        = max(-1.0, min(1.0, new_cinema.get('colorTemp', -0.91)))
    new_cinema['vignette_strength']= max(0.0, min(1.0, new_cinema.get('vignette_strength', 0.84)))
    for key in ['comet_mult', 'supernova_mult', 'globular_density',
                'meteor_frequency', 'grav_ripple_mult', 'dark_cloud_opacity',
                'aurora_mult', 'variable_star_amp', 'cosmos_glow_base']:
        if key in new_cosmos:
            new_cosmos[key] = max(0.1, min(6.0, new_cosmos[key]))

    return new_cinema, new_cosmos


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_cosmos_loop(
    character_slug,
    max_iters=10,
    target_score=9.0,
    api_key='',
    output_root='',
    min_improvement=0.10,
    plateau_rounds=3,
    warm_start='',
):
    if api_key:
        os.environ['ANTHROPIC_API_KEY'] = api_key
    elif not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError("No API key. Pass --key or set ANTHROPIC_API_KEY.")

    from soul_star._api import generate_cosmos_soul_star
    from soul_star._cinema import CINEMA_P

    # Load character spec
    root = Path.home() / 'soul-star'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module(f'soul_star_{character_slug}')
    spec = mod.SPEC

    if not output_root:
        output_root = str(Path.home() / 'Desktop' / 'CosmosLoop')
    base_dir = Path(output_root) / character_slug
    base_dir.mkdir(parents=True, exist_ok=True)

    cinema_p = dict(CINEMA_P)
    cosmos_p = dict(COSMOS_P)

    if warm_start and os.path.exists(warm_start):
        with open(warm_start) as f:
            warm = json.load(f)
        if 'cinema_p' in warm:
            cinema_p.update(warm['cinema_p'])
        if 'cosmos_p' in warm:
            cosmos_p.update(warm['cosmos_p'])
        print(f"  [warm-start] Loaded params from {warm_start}")
        print(f"  [warm-start] Starting score: {warm.get('best_score', '?')}")

    iteration_logs = []
    best_cinema_p = dict(cinema_p)
    best_cosmos_p = dict(cosmos_p)
    best_score = -1.0
    prev_score = -1.0
    plateau_count = 0

    print(f"\n{'='*72}")
    print(f"  COSMOS LOOP — {spec['name']} ({character_slug})")
    print(f"  max_iters={max_iters}, target={target_score}")
    print(f"  Output: {base_dir}")
    print(f"{'='*72}\n")

    for iteration in range(1, max_iters + 1):
        n_frames = 40
        print(f"\n--- Iteration {iteration}/{max_iters} ({n_frames} frames @ 12fps) ---")

        iter_dir = base_dir / f'iter_{iteration:02d}'
        iter_dir.mkdir(parents=True, exist_ok=True)

        render_spec = dict(spec)
        render_spec['n_frames'] = n_frames

        t0 = time.time()
        mp4_path = generate_cosmos_soul_star(
            render_spec, iter_dir, cinema_p=cinema_p, cosmos_p=cosmos_p, fps=12
        )
        print(f"  Rendered in {time.time()-t0:.1f}s → {mp4_path}")

        # Sample frames
        cosmos_dir = iter_dir / '_cosmos_frames'
        all_frames = sorted(cosmos_dir.glob('cosmos_*.png'))
        n = len(all_frames)
        sample_frames = [str(all_frames[i]) for i in sorted({0, n // 2, n - 1}) if i < n]
        print(f"  Sample frames: {[Path(f).name for f in sample_frames]}")

        # Director panel
        print(f"\n  Director panel review (4 directors)...")
        feedbacks = _cosmos_panel_review(spec, cinema_p, cosmos_p, sample_frames, iteration)

        avg_score = sum(fb['score'] for fb in feedbacks) / len(feedbacks)
        print(f"  Average score: {avg_score:.2f}/10")

        if avg_score > best_score:
            best_score = avg_score
            best_cinema_p = dict(cinema_p)
            best_cosmos_p = dict(cosmos_p)

        # Log
        log_entry = {
            'iteration': iteration,
            'avg_score': avg_score,
            'cinema_p': dict(cinema_p),
            'cosmos_p': dict(cosmos_p),
            'feedbacks': feedbacks,
        }
        iteration_logs.append(log_entry)
        log_path = base_dir / 'cosmos_loop_log.json'
        log_path.write_text(json.dumps(iteration_logs, indent=2, ensure_ascii=False))
        print(f"  Log: {log_path}")

        # Stopping conditions
        if avg_score >= target_score:
            print(f"\n  Target {target_score} reached ({avg_score:.2f}). Stopping.")
            break

        if prev_score >= 0:
            improvement = avg_score - prev_score
            if improvement < min_improvement:
                plateau_count += 1
                print(f"  Plateau {plateau_count}/{plateau_rounds} ({improvement:+.2f})")
            else:
                plateau_count = 0
        prev_score = avg_score

        if plateau_count >= plateau_rounds:
            print(f"\n  Plateau detected. Stopping.")
            break

        # Merge and update
        if iteration < max_iters:
            cinema_p, cosmos_p = _merge_cosmos_feedback(feedbacks, cinema_p, cosmos_p)
            print(f"  Params updated for iteration {iteration+1}.")

    # Final 120-frame render with best params
    print(f"\n{'='*72}")
    print("  Final 120-frame cosmos render with best params...")
    final_dir = base_dir / 'final'
    final_dir.mkdir(exist_ok=True)
    final_spec = dict(spec); final_spec['n_frames'] = 120
    t0 = time.time()
    final_mp4 = generate_cosmos_soul_star(
        final_spec, final_dir, cinema_p=best_cinema_p, cosmos_p=best_cosmos_p, fps=12
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Save best params
    best_params_path = base_dir / 'cosmos_best_params.json'
    best_params_path.write_text(json.dumps({
        'best_score': best_score,
        'cinema_p': best_cinema_p,
        'cosmos_p': best_cosmos_p,
    }, indent=2, ensure_ascii=False))

    print(f"\n{'='*72}")
    print(f"  COSMOS LOOP COMPLETE")
    print(f"  Best score: {best_score:.2f}/10")
    print(f"  Final MP4: {final_mp4}")
    print(f"  Best params: {best_params_path}")
    print(f"{'='*72}\n")
    return final_mp4


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Soul Star Cosmos Loop')
    parser.add_argument('character_slug')
    parser.add_argument('--iters',     type=int,   default=10)
    parser.add_argument('--target',    type=float, default=9.0)
    parser.add_argument('--key',       type=str,   default='')
    parser.add_argument('--output',    type=str,   default='')
    parser.add_argument('--min-improvement', type=float, default=0.10)
    parser.add_argument('--plateau-rounds',  type=int,   default=3)
    parser.add_argument('--warm-start',      type=str,   default='',
                        help='Path to cosmos_best_params.json to seed starting params')
    args = parser.parse_args()

    run_cosmos_loop(
        character_slug=args.character_slug,
        max_iters=args.iters,
        target_score=args.target,
        api_key=args.key,
        output_root=args.output,
        min_improvement=args.min_improvement,
        plateau_rounds=args.plateau_rounds,
        warm_start=args.warm_start,
    )


if __name__ == '__main__':
    main()
