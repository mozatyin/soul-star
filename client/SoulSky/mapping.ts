/**
 * SoulSky/mapping.ts
 *
 * Converts SoulGraph pipeline output → SkyRenderParams.
 * Pure functions, no side effects, runs in <1ms.
 */

import type { SoulGraphOutput, SkyRenderParams, NebulaDef, SurfaceStar } from './types'

// ─── Domain metadata ──────────────────────────────────────────────────────────

/** 17 SoulGraph domains with canvas position + star color */
const DOMAIN_META: Record<string, {
  x: number; y: number
  color: string
  label_cn: string
  ring: 'center' | 'inner' | 'middle' | 'outer'
}> = {
  // Center
  EMO: { x: 0.500, y: 0.500, color: '#fffae0', label_cn: '情感',    ring: 'center' },
  // Inner ring (radius ≈ 0.16)
  REL: { x: 0.660, y: 0.500, color: '#ffb0c8', label_cn: '关系',    ring: 'inner' },
  CAR: { x: 0.580, y: 0.361, color: '#ffd080', label_cn: '事业',    ring: 'inner' },
  HEA: { x: 0.420, y: 0.361, color: '#80ffb0', label_cn: '健康',    ring: 'inner' },
  PUR: { x: 0.340, y: 0.500, color: '#d0c0ff', label_cn: '使命',    ring: 'inner' },
  IDE: { x: 0.420, y: 0.639, color: '#80e8ff', label_cn: '身份',    ring: 'inner' },
  MON: { x: 0.580, y: 0.639, color: '#ffec80', label_cn: '财务',    ring: 'inner' },
  // Middle ring (radius ≈ 0.29)
  FAM: { x: 0.739, y: 0.395, color: '#ffb880', label_cn: '家庭',    ring: 'middle' },
  SOC: { x: 0.739, y: 0.605, color: '#b0d0ff', label_cn: '社交',    ring: 'middle' },
  SPI: { x: 0.500, y: 0.212, color: '#c8a8ff', label_cn: '灵性',    ring: 'middle' },
  PHY: { x: 0.261, y: 0.395, color: '#c0ffc8', label_cn: '身体',    ring: 'middle' },
  INT: { x: 0.261, y: 0.605, color: '#c0d8ff', label_cn: '智识',    ring: 'middle' },
  CRE: { x: 0.500, y: 0.788, color: '#ffb0e0', label_cn: '创造',    ring: 'middle' },
  // Outer ring (radius ≈ 0.41)
  LEI: { x: 0.855, y: 0.292, color: '#ffe0a0', label_cn: '休闲',    ring: 'outer' },
  ENV: { x: 0.855, y: 0.708, color: '#a0ffd8', label_cn: '环境',    ring: 'outer' },
  COM: { x: 0.145, y: 0.708, color: '#ffd0a0', label_cn: '社区',    ring: 'outer' },
  EDU: { x: 0.145, y: 0.292, color: '#c0d4ff', label_cn: '学习',    ring: 'outer' },
}

// ─── Challenge → nebula color ──────────────────────────────────────────────────

const CHALLENGE_COLORS: Record<string, [number, number, number]> = {
  loneliness:            [0.25, 0.38, 0.82],
  grief:                 [0.16, 0.10, 0.50],
  anxiety:               [0.88, 0.50, 0.13],
  job_loss:              [0.75, 0.25, 0.13],
  relationship_conflict: [0.82, 0.25, 0.44],
  health_concern:        [0.13, 0.66, 0.50],
  purpose_loss:          [0.44, 0.31, 0.88],
  financial_stress:      [0.75, 0.56, 0.06],
  family_conflict:       [0.82, 0.38, 0.13],
  identity_crisis:       [0.13, 0.69, 0.82],
  trauma:                [0.50, 0.00, 0.13],
  addiction:             [0.63, 0.13, 0.38],
  burnout:               [0.50, 0.31, 0.25],
  longing:               [0.25, 0.50, 0.75],
  anger:                 [0.82, 0.13, 0.00],
  shame:                 [0.38, 0.19, 0.38],
  fear:                  [0.19, 0.25, 0.50],
  isolation:             [0.13, 0.19, 0.38],
}

const DEFAULT_NEBULA_COLOR: [number, number, number] = [0.25, 0.35, 0.70]

// ─── Nebula canvas layout ──────────────────────────────────────────────────────

/** Pre-defined positions for up to 6 nebula slots */
const NEBULA_SLOTS = [
  { cx: 0.38, cy: 0.42 },
  { cx: 0.62, cy: 0.55 },
  { cx: 0.25, cy: 0.65 },
  { cx: 0.70, cy: 0.30 },
  { cx: 0.50, cy: 0.25 },
  { cx: 0.55, cy: 0.75 },
]

// ─── Main mapping function ─────────────────────────────────────────────────────

export function soulGraphToSkyParams(soul: SoulGraphOutput): SkyRenderParams {
  const { domains, challenge_vector, dimensions, traits, intent, match } = soul

  // ── 1. Nebulas from top challenges ─────────────────────────────────────────
  const sortedChallenges = Object.entries(challenge_vector)
    .filter(([, v]) => v > 0.25)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6)

  const nebulas: NebulaDef[] = sortedChallenges.map(([key, intensity], i) => ({
    cx: NEBULA_SLOTS[i].cx,
    cy: NEBULA_SLOTS[i].cy,
    radius: 0.18 + intensity * 0.22,
    color: CHALLENGE_COLORS[key] ?? DEFAULT_NEBULA_COLOR,
    intensity: 0.5 + intensity * 0.5,
    seed: hashStr(key),
    label: key,
  }))

  // ── 2. Surface stars from active domains ───────────────────────────────────
  const stars: SurfaceStar[] = []

  for (const [domain, score] of Object.entries(domains)) {
    if (score < 0.30) continue
    const meta = DOMAIN_META[domain]
    if (!meta) continue

    stars.push({
      x: meta.x,
      y: meta.y,
      magnitude: 2 + score * 8,        // 2–10
      color: meta.color,
      label: meta.label_cn,
      isSoulNode: false,
    })
  }

  // Soul node: urgency-driven bright center star
  if (intent.urgency > 0.1) {
    stars.push({
      x: 0.500,
      y: 0.500,
      magnitude: 6 + intent.urgency * 4,
      color: urgencyColor(intent.urgency, traits.attachment_style),
      label: '当下',
      isSoulNode: true,
    })
  }

  // Guide star (top master match)
  if (match) {
    stars.push({
      x: 0.82,
      y: 0.18,
      magnitude: 7 + match.final_score * 3,
      color: '#ffe8c0',
      label: match.master_name,
      isSoulNode: false,
    })
  }

  // ── 3. Constellation: connect inner-ring domains that are active ───────────
  const constellations = buildConstellations(domains)

  // ── 4. Global rendering params — calibrated from cinema-loop best render ──
  //
  // Base values anchored to director-approved winning params:
  //   bgDensity≈1.70, nebulaGamma≈0.49, colorTemp≈-0.91,
  //   vignetteStrength≈0.84, soulGlow≈1.79
  // at neutral inputs (crisis=0.5, fragility=0.5, attachment=0.5, urgency=0.5).
  // Each dimension shifts its param around the calibrated base.
  const params = {
    // Star density: calm (crisis=0) → dense 2.8, crisis (crisis=1) → sparse 0.6
    bgDensity: 0.6 + (1 - dimensions.crisis_level) * 2.2,

    // Nebula contrast: resilient (fragility=0) → soft 0.25, fragile (fragility=1) → sharp 0.73
    nebulaGamma: 0.25 + dimensions.fragility * 0.48,

    // Temperature: base −0.91 (cinematic cool-blue); secure shifts warmer, insecure cooler
    colorTemp: Math.max(-1.0, Math.min(1.0,
      -0.91 + (dimensions.attachment - 0.5) * 0.6
    )),

    // Vignette: floor 0.50, crisis pushes toward 1.0 (darkness closing in)
    vignetteStrength: Math.min(1.0, 0.50 + dimensions.crisis_level * 0.68),

    // Soul glow: scales with urgency; low=1.0, high=2.58
    soulGlow: 1.0 + intent.urgency * 1.58,

    dominantColor: dominantHex(nebulas),
  }

  return { nebulas, stars, constellations, params }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function hashStr(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0
  return Math.abs(h) % 1000
}

function urgencyColor(urgency: number, attachment: string): string {
  if (urgency > 0.7) return '#ff8060'   // crisis: red-orange
  if (attachment === 'anxious') return '#c0d8ff'  // anxious: cool blue
  if (attachment === 'avoidant') return '#c0ffd8' // avoidant: cool green
  return '#fffae0'                                 // secure: warm white
}

function dominantHex(nebulas: NebulaDef[]): string {
  if (!nebulas.length) return '#3050a0'
  const [r, g, b] = nebulas[0].color
  return `#${Math.round(r * 255).toString(16).padStart(2, '0')}` +
         `${Math.round(g * 255).toString(16).padStart(2, '0')}` +
         `${Math.round(b * 255).toString(16).padStart(2, '0')}`
}

/**
 * Build simple constellations connecting inner-ring domains that are active.
 * Each domain cluster = one constellation group.
 */
function buildConstellations(domains: Record<string, number>) {
  const INNER = ['REL', 'CAR', 'HEA', 'PUR', 'IDE', 'MON']
  const active = INNER.filter(d => (domains[d] ?? 0) > 0.35)
  if (active.length < 2) return []

  const nodes = active.map(d => {
    const meta = DOMAIN_META[d]
    return { x: meta.x, y: meta.y, label: meta.label_cn, isSoul: (domains[d] ?? 0) > 0.70 }
  })

  // Connect adjacent nodes in a ring
  const edges: Array<[number, number]> = nodes.map((_, i) => [i, (i + 1) % nodes.length])

  return [{
    name: '灵魂星座',
    nodes,
    edges,
    color: '#8090c8',
  }]
}
