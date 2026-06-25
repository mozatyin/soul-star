/**
 * SoulSky/types.ts
 *
 * Data types flowing from SoulGraph → visual renderer.
 * SoulGraphOutput mirrors the M1/M2/M3/M4 pipeline outputs.
 */

// ─── SoulGraph pipeline output (server side) ─────────────────────────────────

export interface SoulGraphOutput {
  /** M1: 17 life-domain intensities, 0–1 */
  domains: Record<string, number>

  /** M1: 28 challenge intensities, 0–1 */
  challenge_vector: Record<string, number>

  /** M2: 15 psychological dimensions, 0–1 */
  dimensions: {
    fragility: number       // 0=resilient, 1=fragile
    crisis_level: number    // 0=stable, 1=crisis
    EQ: number              // emotional intelligence
    emotion: number         // current emotional intensity
    attachment: number      // 0=insecure, 1=secure
    [key: string]: number
  }

  /** M2: categorical traits */
  traits: {
    attachment_style: 'secure' | 'anxious' | 'avoidant' | 'disorganized'
    conflict_style: string
    fragility: string
    [key: string]: string
  }

  /** M3: current conversational intent */
  intent: {
    intent_type: string
    urgency: number         // 0–1, drives soul-node glow
    role: 'seeker' | 'giver' | 'both'
  }

  /** M4: top master match (guide star) */
  match?: {
    master_id: string
    master_name: string
    final_score: number
  }
}

// ─── Derived visual data (computed client-side from SoulGraph) ────────────────

export interface NebulaDef {
  cx: number          // 0–1 normalized canvas position
  cy: number
  radius: number      // 0–1 normalized
  color: [number, number, number]   // RGB 0–1
  intensity: number   // 0–1
  seed: number        // deterministic noise offset
  label: string       // e.g. "孤独感"
}

export interface SurfaceStar {
  x: number           // 0–1
  y: number
  magnitude: number   // 1–10
  color: string       // hex
  label: string       // domain / challenge name
  isSoulNode: boolean // true = glowing soul node
}

export interface ConstellationDef {
  name: string
  nodes: Array<{ x: number; y: number; label: string; isSoul: boolean }>
  edges: Array<[number, number]>
  color: string       // hex
}

/** Everything the renderer needs — derived once from SoulGraphOutput */
export interface SkyRenderParams {
  nebulas: NebulaDef[]
  stars: SurfaceStar[]
  constellations: ConstellationDef[]

  /** Global rendering knobs (0–1 unless noted) */
  params: {
    bgDensity: number       // background star count multiplier
    nebulaGamma: number     // contrast: 1.0 = neutral, >1 = darker
    colorTemp: number       // -1 cool → +1 warm
    vignetteStrength: number
    soulGlow: number        // urgency → soul-node glow radius multiplier
    dominantColor: string   // hex — sky tint
  }
}
