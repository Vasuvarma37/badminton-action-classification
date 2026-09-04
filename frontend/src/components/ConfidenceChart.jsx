/**
 * ConfidenceChart.jsx
 * Horizontal bar chart showing softmax scores for all 4 classes.
 */

import { useEffect, useState } from 'react'

const CLASS_LABELS = {
  backhand_drive:    { emoji: '🔙', short: 'Backhand Drive' },
  backhand_net_shot: { emoji: '🎯', short: 'Backhand Net' },
  forehand_clear:    { emoji: '💫', short: 'Forehand Clear' },
  forehand_drive:    { emoji: '⚡', short: 'Forehand Drive' },
}

export default function ConfidenceChart({ classes = [], scores = [], predicted }) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 100)
    return () => clearTimeout(t)
  }, [scores])

  const sorted = [...classes.map((c, i) => ({ cls: c, score: scores[i] ?? 0 }))]
    .sort((a, b) => b.score - a.score)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {sorted.map(({ cls, score }) => {
        const meta = CLASS_LABELS[cls] || { emoji: '🏸', short: cls }
        const pct  = (score * 100).toFixed(1)
        const isTop = cls === predicted

        return (
          <div key={cls}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>{meta.emoji}</span>
                <span style={{
                  fontSize: '0.875rem', fontWeight: isTop ? 700 : 500,
                  color: isTop ? 'var(--accent-green)' : 'var(--text-secondary)',
                }}>
                  {meta.short}
                </span>
                {isTop && <span className="badge badge-green" style={{ fontSize: '0.65rem', padding: '2px 8px' }}>TOP</span>}
              </div>
              <span style={{
                fontSize: '0.875rem', fontWeight: 700, minWidth: 48, textAlign: 'right',
                color: isTop ? 'var(--accent-green)' : 'var(--text-muted)',
              }}>
                {pct}%
              </span>
            </div>

            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: animated ? `${pct}%` : '0%',
                  background: isTop
                    ? 'linear-gradient(90deg, #00ff88, #00d4ff)'
                    : 'linear-gradient(90deg, rgba(0,212,255,0.4), rgba(0,255,136,0.4))',
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
