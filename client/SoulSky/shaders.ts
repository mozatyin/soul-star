/**
 * SoulSky/shaders.ts
 *
 * SkSL (Skia Shader Language) source for the nebula layer.
 *
 * Technique: two-round domain-warped fractal turbulence + dark absorption lanes
 * → produces Hubble/JWST-grade filamentary nebula structure on GPU.
 * Runs in ~2ms on iPhone 12+ via Metal.
 */

export const NEBULA_SHADER = `
uniform float2 iResolution;
uniform int    nebCount;       // active nebulas, max 6

uniform float2 centers[6];    // normalized 0–1 canvas position
uniform float3 colors[6];     // RGB 0–1
uniform float  radii[6];      // normalized radius
uniform float  intensities[6];
uniform float  seeds[6];

uniform float  gamma;          // nebula contrast  (try 1.2–1.6)
uniform float  colorTemp;      // -1 cool → +1 warm

// ── Hash / value noise ─────────────────────────────────────────────────────

float2 hash2(float2 p) {
    p = float2(dot(p, float2(127.1, 311.7)),
               dot(p, float2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

float vnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(dot(hash2(i),               f),
            dot(hash2(i + float2(1,0)), f - float2(1,0)), u.x),
        mix(dot(hash2(i + float2(0,1)), f - float2(0,1)),
            dot(hash2(i + float2(1,1)), f - float2(1,1)), u.x),
        u.y) * 0.5 + 0.5;
}

// ── Fractional Brownian Motion (5 octaves) ─────────────────────────────────

float fbm5(float2 p) {
    float v = 0.0, a = 0.55, f = 1.8;
    for (int i = 0; i < 5; i++) {
        v += a * vnoise(p * f);
        f *= 2.1; a *= 0.48;
    }
    return v;
}

// ── Turbulence — abs-value FBM creates filamentary ISM structure ───────────

float turb(float2 p) {
    float t = 0.0, a = 0.52, f = 1.6;
    for (int i = 0; i < 5; i++) {
        t += a * abs(vnoise(p * f) * 2.0 - 1.0);
        f *= 2.05; a *= 0.50;
    }
    return t;
}

// ── Main ───────────────────────────────────────────────────────────────────

half4 main(float2 fragCoord) {
    float2 uv = fragCoord / iResolution;
    float3 col = float3(0.0);

    for (int n = 0; n < 6; n++) {
        if (n >= nebCount) break;

        float2 cen   = centers[n];
        float  rad   = max(radii[n], 0.01);
        float3 nebCol = colors[n];
        float  inten = intensities[n];
        float  seed  = seeds[n];

        // Radial envelope — fade out from center
        float dist = length((uv - cen) / rad);
        float envelope = exp(-dist * dist * 2.5);
        if (envelope < 0.003) continue;

        // ── Round 1 domain warp ──────────────────────────────────────────
        float2 q = float2(
            fbm5(uv * 3.1 + float2(seed * 0.41 + 1.7, seed * 0.13 + 9.2)),
            fbm5(uv * 3.1 + float2(seed * 0.17 + 8.3, seed * 0.71 + 2.8)));
        float2 wuv = uv * 4.0 + q * 1.15;

        // ── Round 2 domain warp (adds deeper complexity) ─────────────────
        float2 r = float2(
            fbm5(wuv + float2(seed * 0.23 + 4.1, 2.0)),
            fbm5(wuv + float2(seed * 0.59 + 5.1, 7.3)));
        wuv += r * 0.6;

        // ── Filamentary density via turbulence ───────────────────────────
        float density = turb(wuv + float2(seed * 0.37));

        // ── Dark absorption lanes (dust columns — the JWST signature) ────
        float darkLane = smoothstep(0.20, 0.68,
            fbm5(wuv * 1.85 + float2(seed * 0.5 + 3.3)));

        // ── Emission brightening near core ───────────────────────────────
        float emission = 1.0 + 3.2 * envelope * envelope;

        float alpha = pow(max(0.0, density * envelope * inten * darkLane), gamma);

        col += nebCol * emission * alpha;
    }

    // ── Filmic tonemapping (avoids blown-out whites) ───────────────────────
    col = col / (col + float3(0.50));
    col = pow(max(col, float3(0.0)), float3(0.90));

    // ── Color temperature tint ─────────────────────────────────────────────
    float warm = max(0.0,  colorTemp);
    float cool = max(0.0, -colorTemp);
    col.r = min(1.0, col.r + warm * 0.09);
    col.b = min(1.0, col.b + cool * 0.11);
    col.g = min(1.0, col.g + warm * 0.03);

    float lum = dot(col, float3(0.299, 0.587, 0.114));
    return half4(half3(col), half(min(1.0, lum * 1.7)));
}
`

/**
 * Vignette + background-star overlay shader.
 * Draws tens of thousands of sub-pixel background stars via hash functions —
 * no CPU loop needed, runs purely on GPU.
 */
export const STARFIELD_SHADER = `
uniform float2 iResolution;
uniform float  starDensity;   // 0.5–2.0
uniform float  seed;
uniform float  colorTemp;     // -1 cool → +1 warm
uniform float  vigStrength;   // 0–1

float hash(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

half4 main(float2 fragCoord) {
    float2 uv = fragCoord / iResolution;

    // ── Background star field ──────────────────────────────────────────────
    // Tile the canvas into a fine grid; one potential star per cell
    float cellSize = 4.0 / starDensity;
    float2 cell = floor(fragCoord / cellSize);
    float2 cellUV = fract(fragCoord / cellSize);

    float h1 = hash(cell + float2(seed, 0.0));
    float h2 = hash(cell + float2(0.0, seed + 73.1));
    float h3 = hash(cell + float2(seed * 1.3, seed * 0.7));

    // Star exists if h1 > threshold (controls density)
    float starBrightness = 0.0;
    if (h1 > 0.72) {
        float2 starPos = float2(h2, h3);  // sub-cell position
        float d = length(cellUV - starPos);
        float r = 0.04 + h1 * 0.08;      // tiny radius
        float brightness = h1 * smoothstep(r, r * 0.3, d);

        // Color: bluish-white (hot) → yellowish (cool)
        float temp = hash(cell + float2(13.7, seed));
        starBrightness = brightness * (0.55 + temp * 0.45);
    }

    float3 starCol = float3(starBrightness);
    // Temperature tint: hot stars bluer, cool stars yellower
    float starTemp = hash(cell + float2(7.3, seed * 2.1));
    starCol.r = starBrightness * (0.85 + starTemp * 0.25);
    starCol.b = starBrightness * (0.85 + (1.0 - starTemp) * 0.30);

    // Global color temperature
    float warm = max(0.0,  colorTemp);
    float cool = max(0.0, -colorTemp);
    starCol.r = min(1.0, starCol.r + warm * 0.06);
    starCol.b = min(1.0, starCol.b + cool * 0.08);

    // ── Vignette ──────────────────────────────────────────────────────────
    float2 centered = uv - 0.5;
    float vignette = 1.0 - vigStrength * dot(centered, centered) * 3.5;
    vignette = max(0.0, vignette);

    float3 col = starCol * vignette;
    return half4(half3(col), half(starBrightness * vignette));
}
`
