# Soul Star Map — Programmer API

Generate animated soul star maps (GIF or MP4) from a character spec dict.

---

## Quick Start

```python
from soul_star import generate_cosmos_soul_star
from soul_star_harry_potter import SPEC   # or define your own spec

output_mp4 = generate_cosmos_soul_star(SPEC, "/path/to/output/dir")
# → /path/to/output/dir/Harry_Potter_灵魂星图_cosmos.mp4
```

---

## Three Entry Points

| Function | Output | Use when |
|---|---|---|
| `generate_soul_star(spec, output_dir)` | Animated GIF | Quick preview |
| `generate_cinema_soul_star(spec, output_dir, cinema_p, fps)` | 1920×1080 MP4, soul view | Cinema quality |
| `generate_cosmos_soul_star(spec, output_dir, cinema_p, cosmos_p, fps)` | 1920×1080 MP4, cosmos view | **Production output** |

All functions return a `Path` to the generated file.

---

## `generate_cosmos_soul_star` — Production API

```python
generate_cosmos_soul_star(
    spec,                    # character spec dict (see Spec Schema below)
    output_dir,              # str or Path — will be created if missing
    cinema_p=None,           # dict | None — override CINEMA_P_V9 values
    cosmos_p=None,           # dict | None — override COSMOS_P_V9 values
    fps=12,                  # int — video frame rate
) -> Path
```

### Default Parameters (v9, score 7.30/10)

Defaults live in `soul_star.defaults`. Import or override:

```python
from soul_star import CINEMA_P_V9, COSMOS_P_V9

# Override individual values
generate_cosmos_soul_star(
    spec, output_dir,
    cosmos_p={"comet_mult": 4.0},   # only this key changes; rest stay at v9
)
```

**CINEMA_P_V9** (cinema compositing):

| Key | Default | Description |
|---|---|---|
| `cinema_w` / `cinema_h` | 1920 / 1080 | Canvas pixel size |
| `soul_fig_sz` | 10.8 | Soul figure size (inches at DPI=100 → 1080px) |
| `depth_star_counts` | [301, 170, 92, 40] | Stars per parallax layer |
| `depth_alphas` | [0.055, 0.115, 0.223, 0.481] | Layer opacities |
| `depth_radii` | [0.583, 1.239, 2.252, 3.665] | Star radii per layer |
| `lens_flare_intensity` | 0.504 | Lens flare brightness |
| `vignette_strength` | 0.805 | Edge darkening |
| `color_grade_temp` | -0.029 | Color temperature shift |
| `color_grade_saturation` | 1.164 | Saturation multiplier |
| `film_grain` | 0.017 | Film grain amount |

**COSMOS_P_V9** (cosmos phenomena multipliers):

| Key | Default | Range | Controls |
|---|---|---|---|
| `comet_mult` | 6.0 | 0.1–6.0 | Comet brightness/tail |
| `supernova_mult` | 5.053 | 0.1–6.0 | Supernova blast radius |
| `globular_density` | 6.0 | 0.1–6.0 | Globular cluster star count |
| `meteor_frequency` | 2.907 | 0.1–6.0 | Meteor shower rate |
| `grav_ripple_mult` | 6.0 | 0.1–6.0 | Gravitational ripple amplitude |
| `satellite_speed` | 1.0 | 0.1–3.0 | Satellite transit speed |
| `dark_cloud_opacity` | 5.583 | 0.1–6.0 | Dark cloud density |
| `aurora_mult` | 2.726 | 0.1–6.0 | Aurora brightness |
| `variable_star_amp` | 3.252 | 0.1–6.0 | Variable star pulsation |
| `cosmos_glow_base` | 0.716 | 0.1–2.0 | Global glow multiplier |

---

## Spec Schema

```python
SPEC = {
    # ── Identity ──────────────────────────────────────────────────
    "name":           str,   # display name, e.g. "哈利·波特"
    "title":          str,   # file prefix, e.g. "Harry_Potter"
    "gif_name":       str,   # output filename, e.g. "Harry_Potter_灵魂星图_cosmos.mp4"
    "W":              float, # canvas width in world units (default 26.0)
    "H":              float, # canvas height in world units (default 26.0)
    "n_frames":       int,   # animation frames (default 60)
    "gif_duration_ms":int,   # GIF frame delay ms (default 300; ignored for MP4)

    # ── Atmosphere curves — list of (pct 0–100, value 0–1) ────────
    "bg_temp_curve":  [(pct, val), ...],  # background color temperature
    "aurora_curve":   [(pct, val), ...],  # aurora intensity

    # ── Domains (nebulae) ─────────────────────────────────────────
    "domains": [
        {
            "cx": float, "cy": float,      # centre in world coords
            "rx": float, "ry": float,      # radii
            "ang": float,                  # rotation degrees
            "dk": "#rrggbb",               # dark colour
            "mk": "#rrggbb",               # mid colour
            "bk": "#rrggbb",               # bright colour
            "seed": int,                   # RNG seed (deterministic)
            "name": str,                   # display name
        },
        ...
    ],

    # ── Intensity curves (one list per domain/relation/pattern) ───
    "domain_curves":   [[(pct, val), ...], ...],
    "relation_curves": [[(pct, val), ...], ...],
    "pattern_curves":  [[(pct, val), ...], ...],

    # ── Relations (cosmos phenomena) ─────────────────────────────
    "relations": [
        {
            "name":        str,
            "domain":      int,            # index into domains list
            "angle":       float,          # degrees from domain centre
            "color":       "#rrggbb",
            "cosmos_type": str,            # see Cosmos Types below
        },
        ...
    ],

    # ── Constellation patterns ────────────────────────────────────
    "patterns": [
        {
            "tmpl":    str,                # "scorpius"|"cassiopeia"|"leo"|"capricorn"|"crown"
            "name_cn": str,
            "cx": float, "cy": float,     # centre
            "sx": float, "sy": float,     # spread
            "soul": {                     # star_name → (label, color)
                "StarName": ("label", "#rrggbb"),
            },
        },
        ...
    ],

    # ── Ecology elements (background phenomena) ───────────────────
    "ecology": [
        {
            "name":       str,
            "etype":      "star"|"neb"|"const",
            "birth":      float,           # pct when element appears
            "death":      float,           # pct when element disappears
            "curve":      [(pct, val), ...],
            "color":      "#rrggbb",       # for etype="star"
            "excl_r":     float,           # exclusion radius (no overlap)
            "cosmos_type": str | None,     # None = rendered as nebula/star only
            # for etype="neb": dk, mk, bk, rx, ry, ang, seed
            # for etype="const": tmpl, sx, sy, soul
        },
        ...
    ],

    # ── Story beats ───────────────────────────────────────────────
    "story_beats": [(pct, short_name, description), ...],

    # ── Sky event scheduling ──────────────────────────────────────
    "sky_event_beats": {
        "meteor":           {beat_name: weight, ...},
        "satellite":        {beat_name: weight, ...},
        "aurora_surge":     {beat_name: weight, ...},
        "variable_star":    {beat_name: weight, ...},
        "comet":            {beat_name: weight, ...},
        "supernova":        {beat_name: weight, ...},
        "dark_cloud":       {beat_name: weight, ...},
        "globular_cluster": {beat_name: weight, ...},
        "grav_ripple":      {beat_name: weight, ...},
    },

    # ── Optional lifecycle label ───────────────────────────────────
    "lifecycle_label": {
        "name":       str,
        "curve":      [(pct, val), ...],
        "thresholds": (float, float, float),  # tier breakpoints
        "tier_names": (str, str, str, str),
        "color":      "#rrggbb",
    } | None,
}
```

---

## Cosmos Types

| `cosmos_type` | Visual | Typical use |
|---|---|---|
| `"comet"` | Near-parabolic flyby with dust + ion tail | Approaching threat |
| `"supernova"` | 3-phase blast: flash → bloom → shell | Sudden death |
| `"globular_cluster"` | Dense stellar community | Loyal group |
| `"meteor"` | Fast streak shower | Action/speed |
| `"grav_ripple"` | Ripple ring from life/death boundary | Sacrifice |
| `"satellite"` | Steady orbital transit | Loyal companion |
| `"dark_cloud"` | Absorbing dark nebula | Creeping dread |
| `"aurora_surge"` | Sinusoidal curtain of colour | Magic/victory burst |
| `"variable_star"` | Pulsating brightness | Psychic link |

Set `cosmos_type=None` on an ecology element to suppress cosmos rendering (renders as plain star/nebula).

---

## Example — Minimal Spec

```python
from soul_star import generate_cosmos_soul_star

spec = {
    "name": "示例人物", "title": "Example", "gif_name": "Example_cosmos.mp4",
    "W": 26.0, "H": 26.0, "n_frames": 60, "gif_duration_ms": 300,
    "bg_temp_curve": [(0, 0.3), (100, 0.7)],
    "aurora_curve":  [(0, 0.1), (100, 0.3)],
    "domains": [
        {"cx":13,"cy":13,"rx":8,"ry":6,"ang":0,
         "dk":"#060210","mk":"#201040","bk":"#4020a0","seed":1,"name":"核心域"},
    ],
    "domain_curves":   [[(0,6),(50,8),(100,7)]],
    "relation_curves": [[(0,4),(50,9),(100,5)]],
    "pattern_curves":  [],
    "relations": [
        {"name":"引导者","domain":0,"angle":45,"color":"#a0c0ff","cosmos_type":"supernova"},
    ],
    "patterns": [],
    "ecology": [],
    "story_beats": [(0,"开始","起点"),(50,"转折","中段"),(100,"结局","终点")],
    "sky_event_beats": {},
    "lifecycle_label": None,
}

mp4 = generate_cosmos_soul_star(spec, "~/Desktop/output")
print(mp4)
```

---

## Reference Spec

See `/Users/michael/soul-star/soul_star_harry_potter.py` for a full production spec
with 5 domains, 5 relations, 4 constellations, 5 ecology elements, and 46 story beats.

## Canonical Parameters

`/Users/michael/soul-star/CosmosLoop_HP_v9/harry_potter/cosmos_best_params.json`
— director-loop optimised (7.30/10), baked into `soul_star/defaults.py`.
