/**
 * SoulSky/SoulSkyLiveView.tsx
 *
 * Wraps SoulSkyRenderer with three ambient animation layers:
 *
 *   Breathing  — subtle full-canvas luminance pulse (~4.2 s cycle)
 *                 feels like the sky is alive and inhaling
 *   Twinkling  — 22 sparkle stars with independent sine oscillation
 *                 overlaid on top of the static star layer
 *   Meteors    — shooting stars every 8–22 s with glowing head + tail
 *
 * All animation runs on the UI thread via Reanimated worklets.
 * Zero server calls during animation — fully offline.
 *
 * Requires: react-native-reanimated ≥ 3.6, @shopify/react-native-skia ≥ 1.3
 */

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { StyleSheet, View } from 'react-native'
import {
  BlurMask,
  Canvas,
  Circle,
  Fill,
  Group,
  Line,
  vec,
} from '@shopify/react-native-skia'
import {
  Easing,
  runOnJS,
  useDerivedValue,
  useFrameCallback,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated'

import { SoulSkyRenderer } from './SoulSkyRenderer'
import type { SkyRenderParams } from './types'

// ─── Types ────────────────────────────────────────────────────────────────────

interface SparkleConfig {
  x: number; y: number; r: number
  phase: number    // radians 0–2π
  speed: number    // oscillation frequency multiplier
  base: number     // minimum opacity
}

interface MeteorConfig {
  x1: number; y1: number
  x2: number; y2: number
}

export interface SoulSkyLiveViewProps {
  skyParams: SkyRenderParams
  width: number
  height: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

const N_SPARKLES       = 22
const BREATH_PERIOD_MS = 4200     // full inhale-exhale cycle
const METEOR_MIN_MS    = 8_000
const METEOR_MAX_MS    = 22_000
const METEOR_DURATION  = 750      // ms for one meteor
const METEOR_TAIL_LAG  = 0.20     // tail trails head by this fraction

// ─── Deterministic sparkle generator ─────────────────────────────────────────

const rng = (seed: number) =>
  ((Math.sin(seed * 127.1 + 311.7) * 43758.5453) % 1 + 1) % 1

function buildSparkles(w: number, h: number): SparkleConfig[] {
  return Array.from({ length: N_SPARKLES }, (_, i) => ({
    x:     rng(i * 3 + 0) * w,
    y:     rng(i * 3 + 1) * h,
    r:     0.6 + rng(i * 3 + 2) * 1.6,
    phase: rng(i * 7 + 3) * Math.PI * 2,
    speed: 0.30 + rng(i * 5 + 5) * 0.95,
    base:  0.08 + rng(i * 11 + 7) * 0.28,
  }))
}

function spawnMeteor(w: number, h: number): MeteorConfig {
  const x1  = w * (0.05 + Math.random() * 0.55)
  const y1  = h * (0.05 + Math.random() * 0.40)
  const len = 90 + Math.random() * 190
  const ang = Math.PI / 4 + (Math.random() - 0.5) * 0.7
  return { x1, y1, x2: x1 + Math.cos(ang) * len, y2: y1 + Math.sin(ang) * len }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/**
 * Single twinkling sparkle.
 * Each instance has its own useDerivedValue — valid React hook usage
 * because each star is a separate component instance.
 */
function Sparkle({
  cfg,
  clock,
}: {
  cfg: SparkleConfig
  clock: ReturnType<typeof useSharedValue<number>>
}) {
  const opacity = useDerivedValue(
    () => Math.max(0.02, Math.min(0.90,
      cfg.base + 0.38 * Math.sin(clock.value / 1000 * cfg.speed + cfg.phase),
    )),
    [clock],
  )
  return <Circle cx={cfg.x} cy={cfg.y} r={cfg.r} color="#cce4ff" opacity={opacity} />
}

/** Full-canvas breathing overlay — very subtle luminance modulation */
function BreathLayer({ clock }: { clock: ReturnType<typeof useSharedValue<number>> }) {
  const opacity = useDerivedValue(
    () => 0.018 + 0.018 * Math.sin(clock.value / (BREATH_PERIOD_MS / (2 * Math.PI))),
    [clock],
  )
  return <Fill color="#182840" opacity={opacity} />
}

/** Shooting star with glowing head and fading tail */
function Meteor({
  cfg,
  progress,
}: {
  cfg: MeteorConfig
  progress: ReturnType<typeof useSharedValue<number>>
}) {
  const { x1, y1, x2, y2 } = cfg

  const tailP = useDerivedValue(() => ({
    x: x1 + (x2 - x1) * Math.max(0, progress.value - METEOR_TAIL_LAG),
    y: y1 + (y2 - y1) * Math.max(0, progress.value - METEOR_TAIL_LAG),
  }), [progress])

  const headP = useDerivedValue(() => ({
    x: x1 + (x2 - x1) * progress.value,
    y: y1 + (y2 - y1) * progress.value,
  }), [progress])

  const headX = useDerivedValue(() => x1 + (x2 - x1) * progress.value, [progress])
  const headY = useDerivedValue(() => y1 + (y2 - y1) * progress.value, [progress])

  // Fade in quickly, hold, then fade out
  const fade = useDerivedValue(() => {
    const p = progress.value
    if (p < 0.10) return p / 0.10
    if (p > 0.78) return (1 - p) / 0.22
    return 1
  }, [progress])

  return (
    <Group opacity={fade}>
      {/* Streak */}
      <Line p1={tailP} p2={headP} strokeWidth={1.6} color="#ffffff" opacity={0.85} />
      {/* Glowing head */}
      <Circle cx={headX} cy={headY} r={2.2} color="#ffffff">
        <BlurMask blur={3.5} style="normal" />
      </Circle>
      <Circle cx={headX} cy={headY} r={1.0} color="#ffffff" opacity={0.95} />
    </Group>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export function SoulSkyLiveView({ skyParams, width, height }: SoulSkyLiveViewProps) {
  // Stable sparkle configs (deterministic, only rebuild on size change)
  const sparkles = useMemo(() => buildSparkles(width, height), [width, height])

  // Continuous clock in ms — drives breathing and twinkling on UI thread
  const clock = useSharedValue(0)
  useFrameCallback((info) => {
    clock.value = info.timeSinceFirstFrame
  })

  // ── Shooting star state machine ───────────────────────────────────────────
  const [meteor, setMeteor] = useState<MeteorConfig | null>(null)
  const progress = useSharedValue(0)
  const timerRef = useRef<ReturnType<typeof setTimeout>>()

  // Kept in a ref so the closure always has the latest width/height
  const scheduleRef = useRef<() => void>(() => {})
  scheduleRef.current = () => {
    const delay = METEOR_MIN_MS + Math.random() * (METEOR_MAX_MS - METEOR_MIN_MS)
    timerRef.current = setTimeout(() => {
      const cfg = spawnMeteor(width, height)
      setMeteor(cfg)
      progress.value = 0
      progress.value = withTiming(1, {
        duration: METEOR_DURATION,
        easing: Easing.out(Easing.quad),
      }, (done) => {
        if (done) runOnJS(setMeteor)(null)
      })
      scheduleRef.current()   // schedule next
    }, delay)
  }

  useEffect(() => {
    scheduleRef.current()
    return () => clearTimeout(timerRef.current)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <View style={{ width, height }}>
      {/* Layer 1+2+3+4: Static soul sky (starfield + nebulas + constellations + stars) */}
      <SoulSkyRenderer skyParams={skyParams} width={width} height={height} />

      {/* Layer 5: Ambient animation overlay */}
      <Canvas style={StyleSheet.absoluteFill}>
        {/* Breathing pulse */}
        <BreathLayer clock={clock} />

        {/* Twinkling sparkles */}
        {sparkles.map((cfg, i) => (
          <Sparkle key={i} cfg={cfg} clock={clock} />
        ))}

        {/* Shooting star */}
        {meteor && <Meteor cfg={meteor} progress={progress} />}
      </Canvas>
    </View>
  )
}
