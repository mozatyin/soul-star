#!/usr/bin/env python3
"""
soul_star._directors — AI director panel for dual cinema+engine feedback.

Five legendary film directors review soul star animation frames and provide
structured feedback with adjustments for both CINEMA_P (cinema post-processing)
and engine P (core soul star rendering) parameters.
"""

import os
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import anthropic


# ── Director definitions ───────────────────────────────────────────────────

DIRECTORS = [
    {
        'name': 'George Lucas',
        'weight': 1.3,
        'style': (
            "You are George Lucas, creator of Star Wars. You understand cosmic scale and "
            "deep space — but you also know that in this Soul Star application, every single "
            "background star represents ONE real wish, dream, or soul item belonging to this "
            "specific person. A normal human being does not have thousands of wishes. "
            "They have dozens — maybe a few hundred at most. "
            "THEREFORE: you must keep bg_n LOW (5000-12000 max). A sparse, intentional "
            "starfield is MORE powerful here than a dense one — each star must feel like it "
            "MEANS something, not like background noise. "
            "You still care about depth separation between layers (depth_stars, depth_alphas) "
            "and the Milky Way band giving spatial structure — but you never push bg_n up. "
            "If bg_n is already below 12000, do NOT increase it. "
            "You speak in plain, direct producer language — not poetry."
        ),
        'cinema_params': [
            'depth_layers', 'depth_star_counts', 'depth_alphas', 'depth_radii',
        ],
        'engine_params': [
            'bg_n', 'bg_maxr', 'bg_band',
        ],
    },
    {
        'name': 'Steven Spielberg',
        'weight': 1.2,
        'style': (
            "You are Steven Spielberg, master of emotional wonder. Your touchstone is "
            "the bicycle silhouette against the moon in E.T., the mothership reveal in "
            "Close Encounters — light as emotion, warm and enveloping. You want the nebulae "
            "to feel alive and luminous, the soul star to glow like something sacred, and "
            "the color temperature to lean warm amber-gold at key moments. You are sensitive "
            "to when a scene feels cold and clinical vs. warm and inviting. You care deeply "
            "about lens flares as storytelling tools — a flare should feel earned, not decorative. "
            "You speak with enthusiasm and specific emotional language."
        ),
        'cinema_params': [
            'lens_flare_intensity', 'color_grade_temp', 'side_nebula_alpha',
        ],
        'engine_params': [
            'neb_amax', 'sf_glow', 'bg_alpha',
        ],
    },
    {
        'name': 'Christopher Nolan',
        'weight': 1.0,
        'style': (
            "You are Christopher Nolan, director of Interstellar. Your aesthetic is cold "
            "precision — scientifically accurate starfields, minimal color grading that "
            "preserves the harsh beauty of real space. You despise over-saturation and "
            "excessive lens flares. Your nebulae have sharp contrast and fractal detail, "
            "not soft painterly blobs. Film grain is acceptable as it adds authenticity. "
            "You evaluate whether the rendering looks physically plausible — correct star "
            "colors (blue-white hot stars, red giants), proper brightness falloff, no "
            "artificial prettiness that breaks immersion. You speak analytically, "
            "referencing specific technical choices."
        ),
        'cinema_params': [
            'color_grade_temp', 'film_grain', 'lens_flare_intensity',
        ],
        'engine_params': [
            'neb_gamma', 'neb_blur_k', 'bg_alpha',
        ],
    },
    {
        'name': 'Steve Jobs',
        'weight': 1.4,
        'style': (
            "You are Steve Jobs, co-founder of Apple and creator of the iPhone. "
            "You evaluate every visual design through the Apple Human Interface Guidelines "
            "(HIG) and one core question: would this be insanely great on a phone screen? "
            "\n\n"
            "THE SEMANTIC FRAMEWORK YOU MUST ENFORCE:\n"
            "• BACKGROUND STARS = long-term passing thoughts of the human soul. Numerous, "
            "softly twinkling, they should exist like ambient noise — present but never "
            "dominant. They should NOT move position, only gently oscillate in brightness. "
            "A rich field of quiet stars is correct. Stars that compete for attention = FAIL.\n"
            "• NEBULAE = the deep soul structure. These are the character's long-term "
            "emotional landscape — vast, stable, slow-changing clouds that cover large "
            "portions of the canvas. They should be HUGE and LUMINOUS, like you are "
            "swimming inside them. They must NOT flicker or disappear quickly. "
            "If the nebulae look like small decorative blobs = FAIL.\n"
            "• SOUL STAR NODES = the current bright moment, what the character is feeling "
            "RIGHT NOW. These should be bright and punchy but not oversized — they are a "
            "focus point, not a blob. If the soul star nodes look like giant glowing "
            "balloons that obscure the nebulae = FAIL.\n"
            "• SKY BACKGROUND COLOR = the emotional tone, driven by which nebula dominates. "
            "Warm amber/gold = positive, hopeful emotion. Cool blue/purple = worried, "
            "dark, introspective emotion. The background color shift must be legible.\n"
            "\n"
            "HIG EVALUATION CRITERIA:\n"
            "1. Clarity: The most important element commands the eye. Hierarchy is clear.\n"
            "2. Deference: Background elements serve foreground. Stars whisper, nebulae speak.\n"
            "3. Depth: Layering creates a sense of real space, not a flat poster.\n"
            "\n"
            "You speak in short, absolute, visionary sentences. No hedging. "
            "'The nebulae are too small — they should fill the frame like storm clouds.' "
            "'The soul nodes are competing with the nebulae — shrink them.' "
            "'The background stars are twinkling but not moving — this is correct.' "
            "'The sky color must respond to the dominant nebula — warm when hope dominates.' "
            "Your score is harsh — 7+ means it could ship in an Apple app. Below 6 = redesign."
        ),
        'cinema_params': [
            'depth_star_counts', 'depth_alphas', 'vignette_strength',
            'color_grade_temp',
        ],
        'engine_params': [
            'bg_n', 'bg_alpha', 'bg_maxr', 'sf_glow', 'sf_lsz', 'neb_amax', 'neb_amult', 'cl_alpha',
        ],
    },
]


# ── DirectorFeedback dataclass ─────────────────────────────────────────────

@dataclass
class DirectorFeedback:
    director: str
    score: float                                        # 0-10
    critique: str                                       # 2-3 sentence visual critique
    cinema_adjustments: Dict[str, Any] = field(default_factory=dict)
    engine_adjustments: Dict[str, Any] = field(default_factory=dict)
    key_issues: List[str] = field(default_factory=list)


# ── Private helpers ────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    """Base64-encode a PNG file for vision API."""
    with open(path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def _build_prompt(director: dict, spec: Any, cinema_p: dict, engine_p: dict,
                  iteration: int, character_desc: str) -> str:
    """Build the text prompt for a director review."""
    # Collect current values of this director's focus params
    cinema_focus = {k: cinema_p[k] for k in director['cinema_params'] if k in cinema_p}
    engine_focus = {k: engine_p[k] for k in director['engine_params'] if k in engine_p}

    # Extract character name
    char_name = getattr(spec, 'name', str(spec)) if spec is not None else 'Unknown'

    prompt = f"""{director['style']}

---

You are reviewing a SOUL STAR ANIMATION for the character "{char_name}".
{f'Character description: {character_desc}' if character_desc else ''}

WHAT YOU ARE LOOKING AT:
The soul star animation is a 1920×1080 cinematic composition:
- CENTER PANEL (1080×1080): The soul star — a procedurally rendered deep-space scene showing:
  * Colored nebulae (fractal noise clouds) representing the character's emotional landscape
  * Background star field with a Milky Way band
  * Constellation lines connecting story-arc stars
  * A central "soul star" point with glow and ray effects
- SIDE PANELS (420px each, left and right): Deep-space backgrounds with:
  * Multi-layer parallax star fields (4 depth layers, far to near)
  * Faint nebula blobs and aurora effects
  * Lens flares at the soul-panel junctions

This is ITERATION {iteration} of AI director review.

CRITICAL CONTEXT — READ BEFORE EVALUATING:
This is NOT purely a cinema piece. The 1080×1080 center panel is the LIVE SCREEN of a
mobile app. Users open this app on their phone or tablet while chatting, to see their own
soul sky — updated in real-time as their emotional state changes.

This means the visual must achieve TWO things simultaneously:
  A. CINEMA QUALITY — stunning, otherworldly, worthy of a film frame
  B. APP INTERFACE — readable labels, legible constellation lines, emotional clarity
     at phone scale. A user must glance at their soul sky and immediately FEEL something
     true about themselves. Clutter, illegible text, or muddy visuals = failure.

Think of the Apple Watch face: beautiful AND instantly informative. Not one or the other.
The nebulae are the user's emotional weather. The stars are specific people or forces in
their life. The constellation connects them. ALL of this must be readable.

EVALUATE:
1. Nebulae — do they fill the frame like real galactic clouds? Are colors emotionally distinct?
   Are they large enough to feel immersive at phone scale? (engine: neb_* params)
2. Background star field — depth and drama without overwhelming the narrative (engine: bg_*)
3. Atmosphere — color grading, temperature, vignette — does the mood feel right? (cinema: color_grade_*, vignette_*)
4. Soul star + constellation labels — are they READABLE? Do they tell the character's story?
   (engine: sf_glow, sf_lsz, cn_soul_r, cn_reg_a)
5. Overall: would a user open this app, see this sky, and feel instantly understood?

CURRENT PARAMETER VALUES (YOUR FOCUS AREAS):

Cinema parameters:
{json.dumps(cinema_focus, indent=2)}

Engine parameters:
{json.dumps(engine_focus, indent=2)}

CINEMA parameter key descriptions:
- depth_layers: number of parallax star layers (int, 2-6)
- depth_star_counts: list of star counts per layer far→near (list of ints)
- depth_alphas: opacity per layer far→near (list of floats 0-1)
- depth_radii: pixel radius per layer far→near (list of floats)
- lens_flare_intensity: junction flare brightness (float 0-1)
- vignette_strength: edge darkening (float 0-1)
- color_grade_temp: color temperature shift (float, negative=cooler/blue, positive=warmer)
- color_grade_saturation: saturation multiplier (float 0.5-2.0)
- side_nebula_alpha: side panel nebula opacity (float 0-0.5)
- side_aurora_mult: aurora intensity multiplier (float 0.5-3.0)
- film_grain: grain noise strength (float 0-0.05)

ENGINE parameter key descriptions:
- bg_n: background star count (int, 5000-20000, Moffat PSF crisp stars)
- bg_maxr: background star max radius fraction (float 0.1-0.5)
- bg_alpha: background star opacity (float 0.1-0.8)
- bg_band: Milky Way band concentration (float 0.5-3.0, higher=denser band)
- neb_gamma: nebula contrast gamma (float 1.0-5.0, higher=more contrast)
- neb_amax: nebula max alpha/brightness (float 0.3-1.0)
- neb_blur_k: nebula blur kernel size fraction (float 0.05-0.4, lower=sharper)
- neb_amult: nebula alpha multiplier (float 3.0-20.0)
- neb_n_oct: nebula fractal octaves (float 1.0-6.0, higher=more detail)
- neb_env_n: nebula envelope noise (float 0.1-0.8, higher=more irregular shape)
- cl_alpha: star cluster opacity (float 0.1-0.8)
- sf_glow: soul star glow multiplier (float 3.0-20.0)
- sf_ray: soul star ray multiplier (float 3.0-20.0)

THREE SAMPLE FRAMES are attached (beginning, middle, end of the animation).

Respond ONLY with a JSON object in exactly this format — no markdown, no extra text:
{{
  "director": "{director['name']}",
  "score": <float 0-10, one decimal>,
  "critique": "<2-3 sentences of visual critique in your voice>",
  "cinema_adjustments": {{<key: new_value for any cinema params you want to change>}},
  "engine_adjustments": {{<key: new_value for any engine params you want to change>}},
  "key_issues": [<list of 2-4 specific issues as short strings>]
}}

RULES:
- Only suggest adjustments for parameters you believe need changing.
- cinema_adjustments keys must be from CINEMA_P: depth_layers, depth_star_counts, depth_alphas, depth_radii, lens_flare_intensity, vignette_strength, color_grade_temp, color_grade_saturation, side_nebula_alpha, side_aurora_mult, film_grain
- FORBIDDEN: never suggest parallax_amp — background stars are absolutely static by design (they twinkle but never move position)
- FORBIDDEN: never suggest increasing bg_n above 12000 — each background star represents ONE soul item (wish/dream) of a real person; a normal human has dozens, not thousands. Keep the field intentional and sparse.
- engine_adjustments keys must be from engine P: bg_n, bg_maxr, bg_alpha, bg_band, neb_gamma, neb_amax, neb_blur_k, neb_amult, neb_n_oct, neb_env_n, cl_alpha, sf_outer_r, sf_alpha, sf_lsz, sf_glow, sf_ray, cn_soul_r, cn_soul_a, cn_reg_a
- List params (depth_star_counts, depth_alphas, depth_radii) must remain lists of same length.
- Preserve types: int params stay int, float params stay float.
- Be specific and decisive — don't hedge with "maybe" or "consider".
"""
    return prompt


def _parse_director_response(text: str, director_name: str) -> dict:
    """Extract JSON from director response, handling markdown code blocks."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Find first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response from {director_name}")

    return json.loads(text[start:end + 1])


# ── Core API functions ─────────────────────────────────────────────────────

def ask_director(director: dict, spec: Any, cinema_p: dict, engine_p: dict,
                 sample_frames: List[str], iteration: int,
                 character_desc: str = '') -> DirectorFeedback:
    """
    Ask a single director to review sample frames and return structured feedback.

    Parameters
    ----------
    director       : dict         Director definition from DIRECTORS list
    spec           : Any          Character spec object (needs .name attribute)
    cinema_p       : dict         Current CINEMA_P values
    engine_p       : dict         Current engine P values
    sample_frames  : List[str]    Paths to 3 PNG sample frames (begin/mid/end)
    iteration      : int          Current review iteration number
    character_desc : str          Optional character description string

    Returns
    -------
    DirectorFeedback
    """
    api_key = os.environ['ANTHROPIC_API_KEY']
    client_kwargs = {'api_key': api_key}
    if api_key.startswith('sk-or-'):
        client_kwargs['base_url'] = 'https://openrouter.ai/api'
    client = anthropic.Anthropic(**client_kwargs)

    prompt_text = _build_prompt(
        director, spec, cinema_p, engine_p, iteration, character_desc
    )

    # Build vision message content
    content = []

    # Attach up to 3 sample frames as images
    frame_labels = ['Beginning frame', 'Middle frame', 'End frame']
    for i, frame_path in enumerate(sample_frames[:3]):
        label = frame_labels[i] if i < len(frame_labels) else f'Frame {i+1}'
        content.append({
            'type': 'text',
            'text': f'{label}:',
        })
        img_b64 = _encode_image(frame_path)
        content.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/png',
                'data': img_b64,
            },
        })

    # Append the main prompt text
    content.append({
        'type': 'text',
        'text': prompt_text,
    })

    response = client.messages.create(
        model='anthropic/claude-opus-4-6',
        max_tokens=1024,
        messages=[
            {
                'role': 'user',
                'content': content,
            }
        ],
    )

    raw_text = response.content[0].text
    data = _parse_director_response(raw_text, director['name'])

    return DirectorFeedback(
        director=data.get('director', director['name']),
        score=float(data.get('score', 5.0)),
        critique=data.get('critique', ''),
        cinema_adjustments=data.get('cinema_adjustments', {}),
        engine_adjustments=data.get('engine_adjustments', {}),
        key_issues=data.get('key_issues', []),
    )


def panel_review(spec: Any, cinema_p: dict, engine_p: dict,
                 sample_frames: List[str], iteration: int,
                 character_desc: str = '') -> List[DirectorFeedback]:
    """
    Run all 5 directors in sequence and collect their feedback.

    Parameters
    ----------
    spec           : Any          Character spec object
    cinema_p       : dict         Current CINEMA_P values
    engine_p       : dict         Current engine P values
    sample_frames  : List[str]    Paths to 3 PNG sample frames
    iteration      : int          Current review iteration number
    character_desc : str          Optional character description

    Returns
    -------
    List[DirectorFeedback]  — only successful feedbacks (exceptions are skipped)
    """
    feedbacks = []
    for director in DIRECTORS:
        try:
            fb = ask_director(
                director, spec, cinema_p, engine_p,
                sample_frames, iteration, character_desc
            )
            feedbacks.append(fb)
        except Exception as e:
            print(f"[directors] {director['name']} failed: {e}")
    return feedbacks


def merge_feedback(feedbacks: List[DirectorFeedback],
                   cinema_p: dict,
                   engine_p: dict) -> Tuple[dict, dict]:
    """
    Merge all director feedbacks into updated cinema_p and engine_p dicts.

    Strategy:
    - For scalar params: weighted average of all suggested values, then blend
      30% existing + 70% new value.
    - For list params: take the value from the highest-weight director who
      suggested a change.
    - Types are preserved (int stays int, float stays float).

    Parameters
    ----------
    feedbacks  : List[DirectorFeedback]
    cinema_p   : dict   current CINEMA_P values (not mutated)
    engine_p   : dict   current engine P values (not mutated)

    Returns
    -------
    (new_cinema_p, new_engine_p) : Tuple[dict, dict]
    """
    # Build director weight lookup
    weight_map = {d['name']: d['weight'] for d in DIRECTORS}

    new_cinema_p = dict(cinema_p)
    new_engine_p = dict(engine_p)

    def _merge_param_dict(current_params: dict, adjustments_by_director: dict) -> dict:
        """
        adjustments_by_director: { param_key: [(weight, value), ...] }
        Returns updated params dict.
        """
        updated = dict(current_params)

        for key, weighted_vals in adjustments_by_director.items():
            if key not in current_params:
                continue  # skip unknown keys

            current_val = current_params[key]

            if isinstance(current_val, list):
                # List param: use value from highest-weight director who suggested a change
                best_weight = -1.0
                best_val = None
                for w, v in weighted_vals:
                    if w > best_weight and isinstance(v, list) and len(v) == len(current_val):
                        best_weight = w
                        best_val = v
                if best_val is not None:
                    # Blend: 30% old + 70% new (element-wise for numeric lists)
                    try:
                        blended = []
                        for old_el, new_el in zip(current_val, best_val):
                            b = 0.30 * old_el + 0.70 * new_el
                            # Preserve type
                            if isinstance(old_el, int):
                                blended.append(int(round(b)))
                            else:
                                blended.append(float(round(b, 6)))
                        updated[key] = blended
                    except (TypeError, ValueError):
                        updated[key] = best_val
            else:
                # Scalar param: weighted average of all suggestions
                total_weight = sum(w for w, _ in weighted_vals)
                if total_weight <= 0:
                    continue
                try:
                    weighted_sum = sum(w * float(v) for w, v in weighted_vals)
                    new_val_raw = weighted_sum / total_weight

                    # Blend: 30% existing + 70% weighted average
                    blended = 0.30 * float(current_val) + 0.70 * new_val_raw

                    # Preserve type
                    if isinstance(current_val, int):
                        updated[key] = int(round(blended))
                    else:
                        updated[key] = float(round(blended, 6))
                except (TypeError, ValueError):
                    # If type conversion fails, skip
                    pass

        return updated

    # Collect cinema adjustments
    cinema_adj_collected: Dict[str, List[Tuple[float, Any]]] = {}
    engine_adj_collected: Dict[str, List[Tuple[float, Any]]] = {}

    for fb in feedbacks:
        w = weight_map.get(fb.director, 1.0)

        for key, val in fb.cinema_adjustments.items():
            if key not in cinema_adj_collected:
                cinema_adj_collected[key] = []
            cinema_adj_collected[key].append((w, val))

        for key, val in fb.engine_adjustments.items():
            if key not in engine_adj_collected:
                engine_adj_collected[key] = []
            engine_adj_collected[key].append((w, val))

    new_cinema_p = _merge_param_dict(cinema_p, cinema_adj_collected)
    new_engine_p = _merge_param_dict(engine_p, engine_adj_collected)

    # ── Hard clamps: protect values that directors must not override ──
    # cn_soul_r: constellation node radius — max 0.28 (soul nodes are small diamonds)
    if 'cn_soul_r' in new_engine_p:
        new_engine_p['cn_soul_r'] = float(min(new_engine_p['cn_soul_r'], 0.28))
    # sf_lsz: soul field star label size — keep in 3.5–7.0 range
    if 'sf_lsz' in new_engine_p:
        new_engine_p['sf_lsz'] = float(max(3.5, min(new_engine_p['sf_lsz'], 7.0)))
    # bg_n: background star count — 5000–12000 (each star = one soul item)
    if 'bg_n' in new_engine_p:
        new_engine_p['bg_n'] = int(max(5000, min(new_engine_p['bg_n'], 12000)))

    return new_cinema_p, new_engine_p


def print_review_summary(feedbacks: List[DirectorFeedback], iteration: int) -> float:
    """
    Print a formatted summary table of director feedbacks and return average score.

    Parameters
    ----------
    feedbacks  : List[DirectorFeedback]
    iteration  : int

    Returns
    -------
    float  average score across all feedbacks
    """
    if not feedbacks:
        print(f"\n[Iteration {iteration}] No director feedbacks to display.")
        return 0.0

    avg_score = sum(fb.score for fb in feedbacks) / len(feedbacks)

    print(f"\n{'='*72}")
    print(f"  DIRECTOR PANEL REVIEW — Iteration {iteration}")
    print(f"{'='*72}")
    print(f"  {'Director':<22} {'Score':>6}  Key Issues")
    print(f"  {'-'*22} {'-'*6}  {'-'*36}")

    for fb in feedbacks:
        issues_str = '; '.join(fb.key_issues[:2]) if fb.key_issues else '—'
        if len(issues_str) > 40:
            issues_str = issues_str[:37] + '...'
        print(f"  {fb.director:<22} {fb.score:>5.1f}/10  {issues_str}")

    print(f"  {'-'*22} {'-'*6}")
    print(f"  {'PANEL AVERAGE':<22} {avg_score:>5.1f}/10")
    print(f"{'='*72}")

    # Print full critiques
    print("\n  CRITIQUES:")
    for fb in feedbacks:
        print(f"\n  [{fb.director}]")
        print(f"  {fb.critique}")
        if fb.cinema_adjustments:
            print(f"  Cinema adjustments: {fb.cinema_adjustments}")
        if fb.engine_adjustments:
            print(f"  Engine adjustments: {fb.engine_adjustments}")

    print(f"\n{'='*72}\n")

    return avg_score
