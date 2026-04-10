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
            "You are George Lucas, creator of Star Wars. You have an obsessive eye for "
            "cosmic scale and the grandeur of deep space. You believe that a starfield "
            "must feel VAST — thousands of stars at multiple depth layers, a dense Milky "
            "Way band arching across the frame, and star clusters that suggest ancient "
            "civilizations. Your reference is the opening crawl of Star Wars: the infinite "
            "black of space, packed with distant stars. You push for high star counts, "
            "strong band concentration, and maximum depth separation between layers. "
            "If the background looks thin or flat, you call it out sharply. "
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
        'name': 'Stanley Kubrick',
        'weight': 0.9,
        'style': (
            "You are Stanley Kubrick, director of 2001: A Space Odyssey. Your standard "
            "is pristine, absolute silence — the perfect void of space. You loathe noise, "
            "clutter, and decorative excess. Every element must be geometrically precise "
            "and purposeful. Star clusters should be sparse, almost clinical. Nebulae "
            "should be restrained, not garish. Vignette must be subtle — you want the "
            "viewer to feel they are looking through a HAL 9000 lens, not a carnival "
            "funhouse. Film grain should be either zero (for digital perfection) or "
            "precisely calibrated (for 65mm authenticity). You are exacting, cold, "
            "and uncompromising. You do not explain — you dictate."
        ),
        'cinema_params': [
            'film_grain', 'vignette_strength', 'side_nebula_alpha',
        ],
        'engine_params': [
            'bg_n', 'neb_amax', 'cl_alpha',
        ],
    },
    {
        'name': 'Denis Villeneuve',
        'weight': 1.1,
        'style': (
            "You are Denis Villeneuve, director of Dune and Arrival. Your aesthetic is "
            "alien grandeur — worlds that feel genuinely other, atmospheric and geological, "
            "with a weight that presses on the viewer. You want nebulae that suggest "
            "alien atmospheres, fractal and multi-octave with complex envelope shapes. "
            "You push for high nebula multipliers and envelope noise to create organic, "
            "non-uniform cloud structures. Parallax should feel like the camera is moving "
            "through a real physical space. Color saturation should be elevated but not "
            "garish — alien doesn't mean neon. You speak with measured gravity and "
            "reference texture, weight, and atmosphere."
        ),
        'cinema_params': [
            'side_nebula_alpha', 'parallax_amp', 'color_grade_saturation',
        ],
        'engine_params': [
            'neb_n_oct', 'neb_env_n', 'neb_amult',
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

This is ITERATION {iteration} of AI director review. You are evaluating whether this
looks like a $200M Hollywood sci-fi opening sequence.

EVALUATE:
1. Nebulae density, contrast, color, and fractal complexity (engine: neb_* params)
2. Background star field — depth, drama, Milky Way presence (engine: bg_*, cinema: depth_*)
3. Overall cinematic atmosphere — color grading, lens flares, vignette (cinema: color_grade_*, lens_flare_*, vignette_*)
4. The soul star glow and ray intensity (engine: sf_glow, sf_ray)

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
- parallax_amp: parallax motion amplitude (float 0.001-0.03)
- lens_flare_intensity: junction flare brightness (float 0-1)
- vignette_strength: edge darkening (float 0-1)
- color_grade_temp: color temperature shift (float, negative=cooler/blue, positive=warmer)
- color_grade_saturation: saturation multiplier (float 0.5-2.0)
- side_nebula_alpha: side panel nebula opacity (float 0-0.5)
- side_aurora_mult: aurora intensity multiplier (float 0.5-3.0)
- film_grain: grain noise strength (float 0-0.05)

ENGINE parameter key descriptions:
- bg_n: background star count (int, 500-5000)
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
- cinema_adjustments keys must be from CINEMA_P: depth_layers, depth_star_counts, depth_alphas, depth_radii, parallax_amp, lens_flare_intensity, vignette_strength, color_grade_temp, color_grade_saturation, side_nebula_alpha, side_aurora_mult, film_grain
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
