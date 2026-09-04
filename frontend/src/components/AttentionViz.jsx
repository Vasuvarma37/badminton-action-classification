/**
 * AttentionViz.jsx
 * Frame-level attention weight heatmap — shown as 30 colored bars.
 * High attention = neon green, low = muted.
 */

import { useEffect, useRef } from 'react'

export default function AttentionViz({ attention = [], onFrameHover, activeFrame }) {
  if (!attention || attention.length === 0) return null

  const max = Math.max(...attention, 0.001)
  const min = Math.min(...attention)

  return (
    <div>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
        Frame attention weights — brighter = more important for classification
      </p>
      <div style={{
        display: 'flex', gap: 3, alignItems: 'flex-end',
        height: 60, padding: '0 2px',
      }}>
        {attention.map((w, i) => {
          const norm       = (w - min) / (max - min + 1e-6)
          const barHeight  = Math.max(8, norm * 56)
          const isActive   = i === activeFrame
          const opacity    = 0.3 + norm * 0.7

          return (
            <div
              key={i}
              title={`Frame ${i + 1}: attention = ${(w * 100).toFixed(1)}%`}
              onMouseEnter={() => onFrameHover && onFrameHover(i)}
              onMouseLeave={() => onFrameHover && onFrameHover(null)}
              style={{
                flex: 1, minWidth: 4,
                height: barHeight,
                borderRadius: 3,
                background: isActive
                  ? '#00ff88'
                  : `rgba(0, ${Math.round(150 + norm * 105)}, ${Math.round(100 + norm * 155)}, ${opacity})`,
                cursor: 'pointer',
                transition: 'height 0.4s ease, background 0.2s',
                boxShadow: isActive ? '0 0 8px rgba(0,255,136,0.6)' : 'none',
              }}
            />
          )
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Frame 1</span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Frame {attention.length}</span>
      </div>
    </div>
  )
}
