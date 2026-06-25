#!/usr/bin/env python3
"""
client/image_loop.py — Harry Potter Soul Sky image iteration loop.

Renders Harry's soul sky at a specific story moment (default pct=72, 牺牲时刻 —
Harry walking into the Forbidden Forest to die), then asks 5 HP-aware film directors
to score both visual quality AND narrative clarity.

Each iteration:
  1. Render 1080×1080 PNG from RENDER_P + HP soul curves at `pct`
  2. Call 5 directors (Alfonso Cuarón, Spielberg, Nolan, J.K. Rowling, Cameron)
  3. Directors evaluate: 40% visual + 60% narrative (is this clearly a HP story moment?)
  4. Merge feedback → update RENDER_P
  5. Repeat up to max_iters

After the loop:
  - Saves winning params to HPImageLoop/best_params.json
  - Prints translation to React Native SkyRenderParams format

Usage:
    cd ~/soul-star
    source .env && export ANTHROPIC_API_KEY
    python3 client/image_loop.py [--pct 72] [--iters 10] [--out ~/Desktop/HPImageLoop]
"""

import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent   # ~/soul-star
CLIENT_DIR = Path(__file__).resolve().parent          # ~/soul-star/client

# Import rendering primitives from preview_render.py (same directory)
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from preview_render import (                          # noqa: E402
    build_starfield, build_nebulas, draw_star, add_vignette,
    W, H,
)

# ─── Harry Potter soul curves (inlined) ───────────────────────────────────────

# Domain intensity curves: [(pct, value), ...]  value range 3.5-9.8
D0_ORPHAN   = [(0,9.5),(14,8.5),(21,7.0),(35,6.0),(49,7.5),(63,8.5),(70,9.0),(84,7.0),(91,5.5),(100,4.5)]
D1_MAGIC    = [(0,3.5),(14,6.0),(21,7.5),(28,8.0),(42,7.0),(56,8.5),(70,9.0),(84,9.5),(91,9.0),(100,8.5)]
D2_FRIENDS  = [(0,3.5),(21,6.0),(28,7.5),(35,8.0),(42,7.5),(49,6.5),(56,8.5),(63,9.0),(70,9.5),(84,9.0),(91,9.5),(100,9.8)]
D3_DARK     = [(0,3.5),(21,4.0),(28,5.0),(35,8.0),(42,7.5),(49,9.0),(56,8.5),(63,9.0),(70,9.5),(77,9.0),(84,9.5),(91,9.8),(100,3.5)]
D4_LOVE     = [(0,8.5),(14,7.5),(28,6.5),(42,6.0),(56,6.5),(63,7.0),(70,7.5),(77,8.0),(84,9.0),(91,9.8),(95,9.5),(100,9.8)]

# Relation influence curves
R0_HERMIONE = [(0,0.0),(21,5.0),(28,7.5),(35,8.5),(49,7.0),(56,9.0),(63,8.5),(70,9.5),(84,9.0),(91,9.5),(100,9.0)]
R1_RON      = [(0,0.0),(21,6.5),(28,8.0),(42,7.5),(49,5.0),(56,4.0),(63,7.0),(70,8.5),(77,9.0),(84,8.5),(91,9.5),(100,9.0)]
R2_DUMBLE   = [(0,0.0),(14,5.0),(21,7.5),(35,8.5),(42,9.0),(49,9.5),(56,5.0),(63,4.0),(70,3.5),(77,4.0),(84,9.0),(91,8.0),(100,7.5)]
R3_VOLDE    = [(0,4.0),(14,3.5),(21,3.0),(28,4.5),(35,7.5),(42,6.5),(49,9.0),(56,8.0),(63,8.5),(70,9.5),(77,9.0),(84,9.5),(91,9.8),(100,0.0)]
R4_LILY     = [(0,9.5),(14,8.0),(28,7.0),(42,6.5),(56,6.5),(70,7.5),(84,9.0),(91,9.5),(100,9.8)]

STORY_BEATS = {
    0: ('女贞路',    "Privet Drive — cupboard under the stairs"),
    6: ('海格',      "Hagrid arrives — 'You're a wizard, Harry'"),
    14: ('分院',     "The Sorting Hat — Gryffindor"),
    22: ('魔法石',   "Philosopher's Stone — Voldemort hiding in Quirrell"),
    28: ('密室',     "Chamber of Secrets — basilisk and Tom Riddle's diary"),
    34: ('墓地',     "The graveyard — Voldemort reborn using Harry's blood"),
    38: ('神秘部',   "Department of Mysteries — Sirius dies"),
    48: ('邓之死',   "Dumbledore's death — the lighthouse falls"),
    58: ('多比之死', "Dobby dies on the beach — 'such a beautiful place to be with friends'"),
    62: ('绝望谷底', "The lowest point — horcrux hunt, Ron leaves, darkness"),
    70: ('枯林',     "The Forbidden Forest — Harry holds the Resurrection Stone"),
    72: ('牺牲时刻', "The Sacrifice — Harry walks to his death. Love as the ultimate magic."),
    74: ('心灵感应', "Voldemort believes he has won — Harry lies still in his arms"),
    78: ('斯内普记忆', "Snape's memories — 'After all this time?' 'Always.'"),
    80: ('白色车站', "King's Cross Station — between life and death. 'Is this real?' 'Both.'"),
    83: ('佯死苏醒', "Narcissa Malfoy's lie saves Harry — Voldemort's fatal mistake"),
    86: ('最终对决', "Expelliarmus vs Avada Kedavra — the Elder Wand refuses"),
    90: ('大战',     "Battle of Hogwarts — dawn after the longest night"),
    100: ('19年后',  "King's Cross — Albus Severus — all shall be well"),
}


def _interp(curve: list, pct: float) -> float:
    """Linear interpolation on a [(pct, val), ...] curve."""
    if pct <= curve[0][0]:  return float(curve[0][1])
    if pct >= curve[-1][0]: return float(curve[-1][1])
    for i in range(len(curve) - 1):
        p0, v0 = curve[i]; p1, v1 = curve[i + 1]
        if p0 <= pct <= p1:
            t = (pct - p0) / (p1 - p0)
            return float(v0 + t * (v1 - v0))
    return float(curve[-1][1])


def _norm(v: float, lo: float = 3.0, hi: float = 10.0) -> float:
    """Normalize curve value to [0, 1]."""
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def nearest_beat(pct: float) -> Tuple[str, str]:
    """Return (chinese_name, english_desc) of the nearest story beat."""
    items = list(STORY_BEATS.items())
    _, (cn, en) = min(items, key=lambda b: abs(b[0] - pct))
    return cn, en


# ─── Render parameters ────────────────────────────────────────────────────────

# At pct=72 (牺牲时刻): cold, dark, high orphan pain, love as quiet undercurrent
DEFAULT_RENDER_P: Dict[str, Any] = {
    # Global atmosphere
    'bgDensity':      1.25,   # star density
    'nebulaGamma':    0.70,   # lower = brighter, more visible nebulae
    'colorTemp':     -0.40,   # negative = cool blue — this is a dark moment
    'vigStrength':    0.52,   # vignette darkening at edges
    'soulGlow':       1.90,   # Harry's central star glow multiplier
    # Per-nebula intensity multipliers (1.0 = curve-driven, higher = boosted)
    'neb0_intensity': 1.10,   # 孤儿之痛 (orphan pain) — dominant deep blue
    'neb1_intensity': 0.85,   # 魔法天赋 (magic talent) — amber gold
    'neb2_intensity': 0.80,   # 友谊之盾 (friendship) — Gryffindor crimson
    'neb3_intensity': 0.65,   # 黑暗对抗 (Voldemort) — killing curse green
    'neb4_intensity': 0.85,   # 牺牲之爱 (Lily's love) — silver white
    # Layout
    'nebSizeMult':    1.10,   # global nebula size multiplier
    'labelAlpha':     210,    # star label opacity (0-255)
    'constAlpha':     0.42,   # constellation line opacity
}

RENDER_P_DOCS = """
RENDER PARAMETERS — use exactly these key names in your "adjustments":

Global atmosphere:
  bgDensity      float 0.5–2.5   background star density
  nebulaGamma    float 0.4–1.4   nebula brightness (lower=brighter/wider)
  colorTemp      float -1.0–0.5  color temperature (negative=cool/blue)
  vigStrength    float 0.0–0.9   vignette edge darkness
  soulGlow       float 0.8–3.0   Harry's central star glow multiplier

Per-nebula intensity (multiplies the story-curve-driven value):
  neb0_intensity float 0.3–1.5   孤儿之痛 — orphan pain — deep indigo-blue
  neb1_intensity float 0.3–1.5   魔法天赋 — magic talent — amber-gold
  neb2_intensity float 0.3–1.5   友谊之盾 — friendship shield — Gryffindor crimson
  neb3_intensity float 0.3–1.5   黑暗对抗 — dark confrontation — killing-curse green
  neb4_intensity float 0.3–1.5   牺牲之爱 — sacrificial love — silver-white

Layout:
  nebSizeMult    float 0.5–2.0   global nebula size multiplier
  labelAlpha     int   50–255    star label text opacity
  constAlpha     float 0.0–0.9  constellation line opacity
"""


# ─── Scene builder ────────────────────────────────────────────────────────────

def build_hp_scene(pct: float, render_p: Dict[str, Any]):
    """
    Evaluate HP soul curves at `pct`, build nebulas + surface_stars + constellation.

    Returns (nebulas, surface_stars, constellation_pts, const_edges)
    """
    sz = render_p['nebSizeMult']

    # Evaluate curves → [0, 1]
    d = [_norm(_interp(c, pct)) for c in (D0_ORPHAN, D1_MAGIC, D2_FRIENDS, D3_DARK, D4_LOVE)]
    r = [_norm(_interp(c, pct)) for c in (R0_HERMIONE, R1_RON, R2_DUMBLE, R3_VOLDE, R4_LILY)]

    # Intensity = render_p multiplier × (base + curve_fraction × range)
    def neb_int(key, base, scale, curve_val):
        return render_p[key] * (base + curve_val * scale)

    nebulas = [
        # 0: 孤儿之痛 — center-left, dominant deep indigo
        dict(cx=0.48, cy=0.47, rx=0.35*sz, ry=0.30*sz,
             color=(0.16, 0.12, 0.88), seed=601,
             intensity=neb_int('neb0_intensity', 0.55, 0.80, d[0])),
        # 1: 魔法天赋 — right side, amber gold
        dict(cx=0.70, cy=0.54, rx=0.27*sz, ry=0.22*sz,
             color=(0.92, 0.78, 0.12), seed=602,
             intensity=neb_int('neb1_intensity', 0.40, 0.80, d[1])),
        # 2: 友谊之盾 — left side, Gryffindor crimson
        dict(cx=0.28, cy=0.54, rx=0.25*sz, ry=0.21*sz,
             color=(0.90, 0.16, 0.12), seed=603,
             intensity=neb_int('neb2_intensity', 0.38, 0.80, d[2])),
        # 3: 黑暗对抗 — upper, killing-curse green
        dict(cx=0.52, cy=0.28, rx=0.24*sz, ry=0.20*sz,
             color=(0.04, 0.72, 0.10), seed=604,
             intensity=neb_int('neb3_intensity', 0.28, 0.80, d[3])),
        # 4: 牺牲之爱 — lower, silver-white
        dict(cx=0.50, cy=0.73, rx=0.22*sz, ry=0.18*sz,
             color=(0.80, 0.88, 0.98), seed=605,
             intensity=neb_int('neb4_intensity', 0.32, 0.80, d[4])),
    ]

    surface_stars = [
        # Central: Harry himself — soul node
        dict(x=0.50, y=0.50, mag=9.5,          color=(1.00, 0.95, 0.85), label='当下·哈利', soul=True),
        # Companions
        dict(x=0.18, y=0.30, mag=4.8+r[0]*2.5, color=(1.00, 0.50, 0.42), label='赫敏'),
        dict(x=0.82, y=0.30, mag=4.8+r[1]*2.5, color=(0.95, 0.60, 0.20), label='罗恩'),
        dict(x=0.87, y=0.14, mag=5.5+r[2]*2.5, color=(0.75, 0.82, 0.98), label='邓布利多'),
        dict(x=0.13, y=0.14, mag=4.5+r[3]*2.5, color=(0.10, 0.90, 0.22), label='伏地魔'),
        dict(x=0.50, y=0.88, mag=5.8+r[4]*2.5, color=(0.94, 0.97, 1.00), label='莉莉的爱'),
    ]

    # Constellation: Harry ↔ all others
    constellation = [
        (0.50, 0.50),  # 0 Harry
        (0.18, 0.30),  # 1 Hermione
        (0.82, 0.30),  # 2 Ron
        (0.50, 0.88),  # 3 Lily
        (0.87, 0.14),  # 4 Dumbledore
    ]
    const_edges = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4)]

    return nebulas, surface_stars, constellation, const_edges


# ─── Rendering ────────────────────────────────────────────────────────────────

def _draw_constellation(canvas: np.ndarray, pts: list, edges: list, alpha: float) -> None:
    base_color = np.array([0.63, 0.70, 0.86], dtype=np.float32) * alpha
    for (a, b) in edges:
        x1, y1 = int(pts[a][0] * W), int(pts[a][1] * H)
        x2, y2 = int(pts[b][0] * W), int(pts[b][1] * H)
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 2 + 1
        for t in np.linspace(0, 1, steps):
            lx = int(x1 + (x2 - x1) * t)
            ly = int(y1 + (y2 - y1) * t)
            if 0 <= lx < W and 0 <= ly < H:
                canvas[ly, lx, :3] = np.minimum(canvas[ly, lx, :3] + base_color, 1.0)


def _add_labels(pil_img: Image.Image, surface_stars: list, beat_cn: str,
                beat_en: str, pct: float, label_alpha: int) -> None:
    try:
        font_lg = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 26)
        font_md = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 18)
        font_sm = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 14)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    d = ImageDraw.Draw(pil_img)
    a = label_alpha

    for star in surface_stars:
        px, py = int(star['x'] * W), int(star['y'] * H)
        cr, cg, cb = (int(c * 255) for c in star['color'])
        off = 28 if star.get('soul') else 18
        d.text((px + off, py - 10), star['label'], fill=(cr, cg, cb, a), font=font_sm)

    # Title
    d.text((W // 2 - 150, 24), '哈利·波特  灵魂星图', fill=(180, 205, 235, 185), font=font_lg)

    # Story beat
    d.text((W // 2 - 160, 62), f'pct={pct:.0f}  ·  {beat_cn}',
           fill=(150, 175, 215, 165), font=font_md)

    # English subtitle at bottom
    en_short = beat_en[:72] + ('…' if len(beat_en) > 72 else '')
    d.text((30, H - 40), en_short, fill=(120, 145, 185, 145), font=font_sm)

    # Nebula legend (bottom-right)
    legend = [
        ('孤儿之痛', (80,  80, 200)),
        ('魔法天赋', (200, 170,  50)),
        ('友谊之盾', (200,  60,  50)),
        ('黑暗对抗', ( 30, 180,  50)),
        ('牺牲之爱', (190, 215, 250)),
    ]
    for i, (name, col) in enumerate(legend):
        y = H - 42 - i * 18
        d.rectangle([W - 110, y + 2, W - 100, y + 12], fill=(*col, 180))
        d.text((W - 95, y), name, fill=(*col, 160), font=font_sm)


def render_scene(pct: float, render_p: Dict[str, Any]) -> Image.Image:
    """
    Render one HP soul sky PNG. Returns PIL Image.
    """
    beat_cn, beat_en = nearest_beat(pct)
    nebulas, surface_stars, constellation, const_edges = build_hp_scene(pct, render_p)

    print('  [1/5] Starfield …', end=' ', flush=True)
    canvas = build_starfield(
        density=render_p['bgDensity'],
        color_temp=render_p['colorTemp'],
        seed=42,
    )
    print('done')

    print('  [2/5] Nebulae (domain-warp) …', end=' ', flush=True)
    neb = build_nebulas(nebulas, gamma=render_p['nebulaGamma'], color_temp=render_p['colorTemp'])
    neb_rgb, neb_a = neb[:, :, :3], neb[:, :, 3:4]
    canvas = np.clip(1 - (1 - canvas) * (1 - neb_rgb * neb_a), 0, 1)
    print('done')

    print('  [3/5] Vignette + constellation …', end=' ', flush=True)
    add_vignette(canvas, render_p['vigStrength'])
    _draw_constellation(canvas, constellation, const_edges, alpha=render_p['constAlpha'])
    print('done')

    print('  [4/5] Stars …', end=' ', flush=True)
    for star in sorted(surface_stars, key=lambda s: s['mag']):
        draw_star(
            canvas,
            int(star['x'] * W), int(star['y'] * H),
            mag=star['mag'],
            color=star['color'],
            is_soul=star.get('soul', False),
            glow_mult=render_p['soulGlow'] if star.get('soul') else 1.0,
        )
    print('done')

    print('  [5/5] Labels …', end=' ', flush=True)
    pil_img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    _add_labels(pil_img, surface_stars, beat_cn, beat_en, pct, render_p['labelAlpha'])
    print('done')

    return pil_img


# ─── HP-aware directors ───────────────────────────────────────────────────────

HP_DIRECTORS = [
    {
        'name':   'Alfonso Cuarón',
        'weight': 1.4,
        'style': (
            "You are Alfonso Cuarón, director of Harry Potter and the Prisoner of Azkaban "
            "and Gravity. You directed the HP film that transformed the series visually — "
            "grounding it in emotional realism, cold British skies, and textures of loss. "
            "You know Harry's orphan wound intimately. You know that his magic is always "
            "tied to his grief — the Patronus charm requires touching the worst memory. "
            "At this moment (pct=72, Harry walking into the forest to die), you want:\n"
            "• The orphan-pain nebula (indigo-blue) to feel like the empty sky Harry "
            "grew up under — vast, pressing, without warmth\n"
            "• The love nebula (silver-white) to be quietly pervasive, not dramatic — "
            "Lily's protection is invisible until it matters\n"
            "• The killing-curse green to loom in the upper frame like a coming storm\n"
            "• Harry's soul star to glow with dignified isolation — bright but alone\n"
            "You evaluate both cinematic composition AND whether the image communicates "
            "the emotional truth of this specific story moment. "
            "You speak with precise visual language and emotional intelligence."
        ),
    },
    {
        'name':   'Steven Spielberg',
        'weight': 1.2,
        'style': (
            "You are Steven Spielberg, master of emotional wonder. You know Harry Potter — "
            "it is the story of an orphan who finds belonging and then sacrifices himself "
            "for that belonging. This is your territory: E.T. going home, the boy silhouetted "
            "against the moon. At pct=72, Harry has just heard his dead parents' voices and "
            "is walking toward death. This must feel beautiful AND heartbreaking simultaneously. "
            "What you want to see:\n"
            "• The warm stars of friendship (Hermione, Ron) must be luminous — they are "
            "still there, still glowing, not knowing Harry is saying goodbye\n"
            "• Lily's star (bottom, silver-white) should carry the scene's emotional weight\n"
            "• Harry's center star must feel solitary — bright but isolated\n"
            "• The overall image should feel like a sacred, quiet moment — not explosive\n"
            "You score warmth, emotional legibility, and whether the image moves you. "
            "You speak with enthusiasm and specific emotional language."
        ),
    },
    {
        'name':   'Christopher Nolan',
        'weight': 1.0,
        'style': (
            "You are Christopher Nolan, director of Interstellar. You see Harry Potter as a "
            "story about love as a physical force that transcends spacetime — exactly as in "
            "your films. At pct=72, Harry is activating ancient blood magic that will turn his "
            "death into a protection field for every living person at Hogwarts. This is "
            "scientifically exact: sacrifice → binding magic → shield. "
            "What you demand:\n"
            "• The orphan-pain nebula (blue-indigo) must dominate because it is the oldest, "
            "deepest trauma — it should have sharp fractal filaments, not soft blobs\n"
            "• Voldemort's green should have harder, sharper edges — dark matter, not gas\n"
            "• Lily's silver nebula should be faint but pervasive, like background radiation\n"
            "• The constellation connecting Harry to all others must be clearly readable\n"
            "• Harry's soul star must have physically correct diffraction spikes\n"
            "You evaluate scientific plausibility, compositional precision, and narrative logic. "
            "You speak analytically and reference specific technical observations."
        ),
    },
    {
        'name':   'J.K. Rowling',
        'weight': 1.5,
        'style': (
            "You are J.K. Rowling, author of Harry Potter. You know every detail of this story — "
            "you wrote it. At pct=72 (牺牲时刻), Harry has just used the Resurrection Stone to "
            "summon the ghosts of his parents, Sirius, and Remus. He let the Stone fall. "
            "He is walking into the Forbidden Forest to let Voldemort's Killing Curse hit him, "
            "knowing that the Horcrux inside him — the piece of Voldemort's soul — will die, "
            "and that Lily's ancient sacrifice magic will activate again: his willing death "
            "will protect every person at Hogwarts. He is 17 years old.\n\n"
            "NARRATIVE ACCURACY CHECKLIST — score each (present/absent/wrong):\n"
            "□ The orphan-pain nebula (indigo-blue) should be DOMINANT — no family to say goodbye to\n"
            "□ Harry's center star should feel ISOLATED and walking, not triumphant\n"
            "□ Voldemort's green nebula should LOOM — it is close, it is coming\n"
            "□ Lily's love nebula (silver-white, lower area) should feel like a QUIET SHIELD\n"
            "□ Hermione and Ron's warm stars should glow BEHIND Harry, not knowing\n"
            "□ Dumbledore's star (upper right) should be BRIGHT — he designed this moment\n"
            "□ The image should feel like standing BETWEEN worlds — not pure darkness, not light\n\n"
            "Your score weights: 70% narrative accuracy + 30% visual quality. "
            "A visually stunning image that misrepresents the story gets a LOW score from you. "
            "You speak as a storyteller — refer to specific book scenes and chapter titles."
        ),
    },
    {
        'name':   'James Cameron',
        'weight': 0.9,
        'style': (
            "You are James Cameron, director of Avatar and Titanic. You know Harry Potter — "
            "you see it as world-building cinema where scale matters. At this moment, Harry's "
            "soul sky should feel OPERATIC: vast nebulae filling the frame like real galactic "
            "clouds, not small decorative patches. "
            "Your primary complaints will be:\n"
            "• Nebulae too small — they should dominate like storm clouds, not float as blobs\n"
            "• Vignette too weak — the frame edges should feel like deep space pressing in\n"
            "• Soul star too dim or too similar to background stars — it must be unmistakable\n"
            "• Starfield lacks depth — should feel like looking through 10,000 light-years\n"
            "What you want: maximum emotional impact per pixel. "
            "The moment Harry walks into that forest must feel COSMIC and INEVITABLE. "
            "You push for bigger, bolder, more immersive — while keeping it legible. "
            "You speak in direct, large-scale cinematographic language."
        ),
    },
]


# ─── Director feedback dataclass ──────────────────────────────────────────────

@dataclass
class HPFeedback:
    director:        str
    score:           float     # weighted: 40% visual + 60% narrative
    visual_score:    float
    narrative_score: float
    critique:        str
    adjustments:     Dict[str, Any] = field(default_factory=dict)
    key_issues:      List[str]      = field(default_factory=list)


# ─── Director API calls ───────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def _parse_json(text: str, director_name: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'No JSON in {director_name} response')
    return json.loads(text[start:end + 1])


def _build_prompt(director: dict, render_p: dict, iteration: int,
                  beat_cn: str, beat_en: str) -> str:
    return f"""{director['style']}

---

STORY MOMENT BEING VISUALIZED:
  {beat_cn} — {beat_en}

This is ITERATION {iteration} of a director feedback loop. Your feedback will update the
rendering parameters. Previous iterations have already been applied to this image.

THE IMAGE shows Harry Potter's Soul Sky — a 1080×1080 cosmic visualization of his soul
at this exact story moment. Every visual element maps to a specific story element:

NEBULAE (colored gas clouds — represent Harry's core emotional landscape):
  • Center: 孤儿之痛 (orphan's pain) — deep indigo-blue — his core wound
  • Right:  魔法天赋 (magical gift) — amber-gold — his power, at peak now
  • Left:   友谊之盾 (friendship shield) — Gryffindor crimson — Hermione + Ron
  • Upper:  黑暗对抗 (dark confrontation) — killing-curse green — Voldemort approaching
  • Lower:  牺牲之爱 (sacrificial love) — silver-white — Lily's ancient magic

STARS (bright points — represent people and forces):
  • Center (soul star): 当下·哈利 Harry himself — isolated, walking toward death
  • Upper-right: 邓布利多 Dumbledore — the guide who engineered this sacrifice
  • Lower-left: 赫敏 Hermione — still fighting at Hogwarts, doesn't know
  • Lower-right: 罗恩 Ron — same
  • Far upper-left: 伏地魔 Voldemort — waiting in the forest
  • Bottom: 莉莉的爱 Lily's love — the origin of everything

CONSTELLATION LINES: Connect Harry to all others — he is connected but walking alone.

CURRENT RENDER PARAMETERS:
{json.dumps(render_p, indent=2)}

{RENDER_P_DOCS}

EVALUATE — give separate scores for:
1. Visual Quality (0-10): Does this look like a real cosmic scene? Nebulae feel like
   galactic clouds? Stars feel physically real? Space feels deep?
2. Narrative Clarity (0-10): Can a Harry Potter reader immediately understand which
   story moment this is? Do the colors/sizes/positions match the emotional reality?

Your FINAL SCORE = 0.4 × visual + 0.6 × narrative.

Respond ONLY with a valid JSON object — no markdown, no preamble:
{{
  "director": "{director['name']}",
  "visual_score": <float 0-10>,
  "narrative_score": <float 0-10>,
  "score": <float 0-10, your weighted average>,
  "critique": "<4-5 sentences addressing BOTH visual quality AND narrative accuracy>",
  "adjustments": {{<key: new_value for parameters you want changed — use exact keys from RENDER PARAMETERS>}},
  "key_issues": ["<issue 1>", "<issue 2>", "<issue 3>"]
}}
"""


def ask_director(director: dict, render_p: dict, image_path: str,
                 iteration: int, beat_cn: str, beat_en: str) -> HPFeedback:
    try:
        import anthropic as _anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        raise EnvironmentError('ANTHROPIC_API_KEY not set')

    kwargs = {'api_key': api_key}
    model  = 'claude-opus-4-6'
    if api_key.startswith('sk-or-'):
        kwargs['base_url'] = 'https://openrouter.ai/api'
        model = 'anthropic/claude-opus-4-6'

    client = _anthropic.Anthropic(**kwargs)

    prompt = _build_prompt(director, render_p, iteration, beat_cn, beat_en)
    img_b64 = _encode_image(image_path)

    content = [
        {'type': 'text',  'text': 'Soul Sky image for evaluation:'},
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64}},
        {'type': 'text',  'text': prompt},
    ]

    resp = client.messages.create(
        model=model, max_tokens=1024,
        messages=[{'role': 'user', 'content': content}],
    )

    data = _parse_json(resp.content[0].text, director['name'])
    return HPFeedback(
        director=data.get('director', director['name']),
        score=float(data.get('score', 5.0)),
        visual_score=float(data.get('visual_score', 5.0)),
        narrative_score=float(data.get('narrative_score', 5.0)),
        critique=data.get('critique', ''),
        adjustments=data.get('adjustments', {}),
        key_issues=data.get('key_issues', []),
    )


def panel_review(render_p: dict, image_path: str, iteration: int,
                 beat_cn: str, beat_en: str) -> List[HPFeedback]:
    feedbacks = []
    for director in HP_DIRECTORS:
        print(f'    → {director["name"]:20s}', end=' ', flush=True)
        try:
            fb = ask_director(director, render_p, image_path, iteration, beat_cn, beat_en)
            feedbacks.append(fb)
            print(f'{fb.score:.1f}  (v:{fb.visual_score:.1f}  n:{fb.narrative_score:.1f})')
        except Exception as e:
            print(f'ERROR: {e}')
    return feedbacks


def print_summary(feedbacks: List[HPFeedback], iteration: int) -> float:
    if not feedbacks:
        return 0.0
    w_map = {d['name']: d['weight'] for d in HP_DIRECTORS}
    total_w = sum(w_map.get(fb.director, 1.0) for fb in feedbacks)
    avg = sum(fb.score * w_map.get(fb.director, 1.0) for fb in feedbacks) / total_w

    print(f'\n  ┌─ Iteration {iteration} · weighted avg {avg:.2f}/10 ──────────────────')
    for fb in feedbacks:
        w = w_map.get(fb.director, 1.0)
        print(f'  │  {fb.director:20s}  {fb.score:.1f}  v:{fb.visual_score:.1f} n:{fb.narrative_score:.1f}  ×{w}')
    print('  │')
    print(f'  │  Critique ({feedbacks[0].director if feedbacks else ""}):')
    if feedbacks:
        for line in feedbacks[0].critique.split('. '):
            if line.strip():
                print(f'  │    {line.strip()}.')
    print('  └──────────────────────────────────────────────────────────')
    return avg


# ─── Feedback merger ──────────────────────────────────────────────────────────

def merge_feedback(feedbacks: List[HPFeedback], render_p: dict) -> dict:
    """
    Weighted merge of all director adjustments → updated render_p.
    Strategy: weighted average of suggested values, then 35% old + 65% new.
    """
    w_map = {d['name']: d['weight'] for d in HP_DIRECTORS}
    suggestions: Dict[str, List[Tuple[float, Any]]] = {}

    for fb in feedbacks:
        w = w_map.get(fb.director, 1.0)
        for k, v in fb.adjustments.items():
            if k in render_p and isinstance(v, (int, float)):
                suggestions.setdefault(k, []).append((w, float(v)))

    new_p = dict(render_p)
    for k, wv_list in suggestions.items():
        current = render_p[k]
        total_w = sum(w for w, _ in wv_list)
        weighted_new = sum(w * v for w, v in wv_list) / total_w
        blended = 0.35 * float(current) + 0.65 * weighted_new
        if isinstance(current, int):
            new_p[k] = int(round(blended))
        else:
            new_p[k] = round(blended, 4)

    return new_p


# ─── Bake winning params back to React Native ─────────────────────────────────

def bake_to_rn(best_p: dict, out_dir: Path) -> None:
    """
    Save winning params as JSON calibration file + print the RN SkyRenderParams equivalent.
    """
    cal = {
        'story_moment': 'pct=72 · 牺牲时刻 · The Sacrifice',
        'render_p': best_p,
        'rn_sky_render_params': {
            'params': {
                'bgDensity':        best_p['bgDensity'],
                'nebulaGamma':      best_p['nebulaGamma'],
                'colorTemp':        best_p['colorTemp'],
                'vignetteStrength': best_p['vigStrength'],
                'soulGlow':         best_p['soulGlow'],
            },
        },
        'note': (
            'Apply rn_sky_render_params.params as the base offsets in '
            'soulGraphToSkyParams() mapping.ts. '
            'The bgDensity floor should be 0.6 + (1-crisis)*range with range '
            f'calibrated so neutral crisis_level=0.5 gives bgDensity≈{best_p["bgDensity"]:.2f}.'
        ),
    }

    cal_path = out_dir / 'best_params.json'
    cal_path.write_text(json.dumps(cal, indent=2, ensure_ascii=False))

    print(f'\n  Best params saved: {cal_path}')
    print('\n  React Native SkyRenderParams equivalent:')
    print('  ─────────────────────────────────────────')
    rn = cal['rn_sky_render_params']['params']
    for k, v in rn.items():
        print(f'    {k}: {v}')
    print('  ─────────────────────────────────────────')
    print('  Nebula intensity calibration for HP 牺牲时刻:')
    for i in range(5):
        k = f'neb{i}_intensity'
        neb_labels = ['孤儿之痛', '魔法天赋', '友谊之盾', '黑暗对抗', '牺牲之爱']
        print(f'    {k} = {best_p[k]:.3f}  ({neb_labels[i]})')


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_image_loop(
    pct:       float = 72.0,
    max_iters: int   = 10,
    target:    float = 8.5,
    out_root:  str   = '',
) -> None:

    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise EnvironmentError(
            'ANTHROPIC_API_KEY not set.\n'
            'Run: source .env && export ANTHROPIC_API_KEY'
        )

    out_dir = Path(out_root or Path.home() / 'Desktop' / 'HPImageLoop')
    out_dir.mkdir(parents=True, exist_ok=True)

    beat_cn, beat_en = nearest_beat(pct)
    render_p  = dict(DEFAULT_RENDER_P)
    best_p    = dict(render_p)
    best_img  = None
    best_score = -1.0
    logs: List[dict] = []

    print(f'\n{"="*72}')
    print(f'  HP SOUL SKY — IMAGE ITERATION LOOP')
    print(f'  Moment: pct={pct:.0f}  ·  {beat_cn}')
    print(f'  "{beat_en}"')
    print(f'  Target: {target}/10   Max iters: {max_iters}')
    print(f'  Output: {out_dir}')
    print(f'{"="*72}\n')

    for it in range(1, max_iters + 1):
        print(f'\n━━━ Iteration {it}/{max_iters} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

        # 1. Render
        t0 = time.time()
        img = render_scene(pct, render_p)
        png_path = out_dir / f'iter_{it:02d}.png'
        img.save(png_path)
        print(f'  → Saved: {png_path}  ({time.time()-t0:.1f}s)')

        # 2. Director panel
        print(f'\n  Director panel:')
        feedbacks = panel_review(render_p, str(png_path), it, beat_cn, beat_en)
        if not feedbacks:
            print('  No valid feedback.')
            continue

        avg = print_summary(feedbacks, it)

        # 3. Track best
        if avg > best_score:
            best_score = avg
            best_p     = dict(render_p)
            best_img   = png_path
            print(f'\n  ★ New best: {avg:.2f}/10')

        # 4. Log
        logs.append({
            'iteration': it, 'pct': pct, 'avg_score': avg,
            'render_p':  dict(render_p),
            'feedbacks': [
                {'director': fb.director, 'score': fb.score,
                 'visual': fb.visual_score, 'narrative': fb.narrative_score,
                 'critique': fb.critique, 'adjustments': fb.adjustments,
                 'issues': fb.key_issues}
                for fb in feedbacks
            ],
        })
        (out_dir / 'image_loop_log.json').write_text(
            json.dumps(logs, indent=2, ensure_ascii=False))

        # 5. Stop if target reached
        if avg >= target:
            print(f'\n  Target {target} reached! Stopping.')
            break

        # 6. Merge feedback → update render_p
        if it < max_iters:
            new_p = merge_feedback(feedbacks, render_p)
            changed = {k: new_p[k] for k in new_p if abs(float(new_p[k]) - float(render_p[k])) > 0.001}
            if changed:
                print(f'\n  Param updates for iteration {it+1}:')
                for k, v in changed.items():
                    print(f'    {k}: {render_p[k]} → {v}')
            render_p = new_p

    # ── After loop ────────────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  LOOP COMPLETE')
    print(f'  Best score: {best_score:.2f}/10   Best image: {best_img}')

    if best_img:
        final = out_dir / 'best.png'
        shutil.copy(str(best_img), str(final))
        print(f'  Best copy: {final}')

    bake_to_rn(best_p, out_dir)
    print(f'{"="*72}\n')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='HP Soul Sky — director-guided image iteration loop',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--pct',   type=float, default=72.0,
                    help='Story moment 0-100 (default 72 = 牺牲时刻)')
    ap.add_argument('--iters', type=int,   default=10,
                    help='Max iterations (default 10)')
    ap.add_argument('--target',type=float, default=8.5,
                    help='Stop when panel avg reaches this score (default 8.5)')
    ap.add_argument('--out',   type=str,   default='',
                    help='Output directory (default ~/Desktop/HPImageLoop)')
    args = ap.parse_args()

    run_image_loop(
        pct=args.pct,
        max_iters=args.iters,
        target=args.target,
        out_root=args.out,
    )


if __name__ == '__main__':
    main()
