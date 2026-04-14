"""
Soul Star Map — canonical v9 rendering parameters (score 7.30/10).

These are the production defaults for generate_cosmos_soul_star().
Pass cinema_p= or cosmos_p= overrides to adjust individual values.
"""

# ── Cinema (1920×1080 compositing) ───────────────────────────────────
CINEMA_P_V9 = {
    "cinema_w":             1920,
    "cinema_h":             1080,
    "soul_fig_sz":          10.8,       # 1080px at DPI=100
    "depth_layers":         5,
    "depth_star_counts":    [301, 170, 92, 40],
    "depth_alphas":         [0.0548, 0.1148, 0.2227, 0.4805],
    "depth_radii":          [0.583, 1.239, 2.252, 3.665],
    "parallax_amp":         0.0,
    "lens_flare_intensity": 0.504,
    "vignette_strength":    0.8046,
    "color_grade_temp":    -0.029,
    "color_grade_saturation": 1.164,
    "side_nebula_alpha":    0.237,
    "side_aurora_mult":     1.395,
    "film_grain":           0.017,
    "bgDensity":            3.914,
    "nebulaGamma":          0.306,
    "colorTemp":           -0.108,
}

# ── Cosmos phenomena multipliers ──────────────────────────────────────
COSMOS_P_V9 = {
    "comet_mult":         6.0,
    "supernova_mult":     5.053,
    "globular_density":   6.0,
    "meteor_frequency":   2.907,
    "grav_ripple_mult":   6.0,
    "satellite_speed":    1.0,
    "dark_cloud_opacity": 5.583,
    "aurora_mult":        2.726,
    "variable_star_amp":  3.252,
    "cosmos_glow_base":   0.716,
    "label_alpha":        0.72,
}
