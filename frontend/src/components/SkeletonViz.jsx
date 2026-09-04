/**
 * SkeletonViz.jsx
 *
 * Renders the 13-joint skeleton overlay on a <canvas> that is always
 * sized to match the actual displayed video element dimensions.
 *
 * The keypoints returned by the API are normalized [0, 1] coordinates
 * (relative to the original video frame). We scale them to the canvas
 * dimensions on every render, so the skeleton stays accurate at any
 * screen size or when the user resizes the window.
 *
 * Joint order (common-13):
 *   0 L-shoulder  1 R-shoulder  2 L-elbow   3 R-elbow
 *   4 L-wrist     5 R-wrist     6 L-hip      7 R-hip
 *   8 L-knee      9 R-knee     10 L-ankle   11 R-ankle
 *  12 nose
 */

import { useRef, useEffect, useCallback } from 'react'

// Skeleton edges: pairs of joint indices to connect with lines
const SKELETON_EDGES = [
  [12, 0], [12, 1],   // head → shoulders
  [0,  1],            // shoulder bar
  [0,  2], [2,  4],   // L arm
  [1,  3], [3,  5],   // R arm
  [0,  6], [1,  7],   // torso sides
  [6,  7],            // hip bar
  [6,  8], [8,  10],  // L leg
  [7,  9], [9,  11],  // R leg
]

// Per-joint colors
const JOINT_COLORS = [
  '#00ff88','#00ff88',  // shoulders
  '#00d4ff','#00d4ff',  // elbows
  '#f59e0b','#f59e0b',  // wrists
  '#a78bfa','#a78bfa',  // hips
  '#00d4ff','#00d4ff',  // knees
  '#00ff88','#00ff88',  // ankles
  '#ffffff',            // nose
]

export default function SkeletonViz({ keypoints, frameIndex = 0, videoRef, style = {} }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // ── Step 1: Size canvas to match the video element's displayed area ──
    const video = videoRef?.current
    let displayW, displayH

    if (video) {
      const rect = video.getBoundingClientRect()
      displayW = rect.width
      displayH = rect.height
    } else {
      const parent = canvas.parentElement
      displayW = parent ? parent.clientWidth  : 640
      displayH = parent ? parent.clientHeight : 360
    }

    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width  = displayW
      canvas.height = displayH
    }

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, displayW, displayH)

    if (!keypoints || keypoints.length === 0) return

    // ── Step 2: Pick the frame ──
    const totalFrames = keypoints.length
    const idx = Math.max(0, Math.min(frameIndex, totalFrames - 1))
    const frame = keypoints[idx]          // array of 26 values: [x0,y0, x1,y1, ...]
    if (!frame || frame.length < 26) return

    // ── Step 3: Parse joints — normalized [0,1] → canvas pixels ──
    const joints = []
    for (let j = 0; j < 13; j++) {
      const nx = frame[j * 2]
      const ny = frame[j * 2 + 1]
      joints.push({ x: nx * displayW, y: ny * displayH })
    }

    // ── Step 4: Draw edges ──
    ctx.lineCap  = 'round'
    ctx.lineJoin = 'round'

    SKELETON_EDGES.forEach(([a, b]) => {
      const jA = joints[a]
      const jB = joints[b]
      if (!jA || !jB) return
      if (jA.x === 0 && jA.y === 0) return   // missing detection
      if (jB.x === 0 && jB.y === 0) return

      ctx.beginPath()
      ctx.moveTo(jA.x, jA.y)
      ctx.lineTo(jB.x, jB.y)
      ctx.strokeStyle = 'rgba(0,212,255,0.65)'
      ctx.lineWidth   = Math.max(2, displayW * 0.004)
      ctx.stroke()
    })

    // ── Step 5: Draw joints ──
    const dotR = Math.max(4, displayW * 0.008)
    joints.forEach((j, idx) => {
      if (j.x === 0 && j.y === 0) return   // missing detection
      const color = JOINT_COLORS[idx] || '#ffffff'

      // Outer glow
      ctx.beginPath()
      ctx.arc(j.x, j.y, dotR * 1.8, 0, Math.PI * 2)
      ctx.fillStyle = color.replace(')', ',0.25)').replace('rgb', 'rgba').replace('#', 'rgba(').replace(/(.{2})(.{2})(.{2})/, (_, r, g, b) =>
        `${parseInt(r,16)},${parseInt(g,16)},${parseInt(b,16)},0.25)`)
      // Simple glow with shadow
      ctx.shadowColor = color
      ctx.shadowBlur  = dotR * 3
      ctx.fill()
      ctx.shadowBlur = 0

      // Core dot
      ctx.beginPath()
      ctx.arc(j.x, j.y, dotR, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    })
  }, [keypoints, frameIndex, videoRef])

  // Redraw whenever keypoints/frameIndex change or on resize
  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(draw)
    })

    const video = videoRef?.current
    if (video) observer.observe(video)
    else if (canvasRef.current?.parentElement) observer.observe(canvasRef.current.parentElement)

    window.addEventListener('resize', draw)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', draw)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [draw, videoRef])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        ...style,
      }}
    />
  )
}
