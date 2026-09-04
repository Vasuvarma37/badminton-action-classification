import { useCallback, useState } from 'react'

const ALLOWED_TYPES = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska']
const ALLOWED_EXTS  = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

export default function VideoUploader({ onFile, disabled }) {
  const [dragOver, setDragOver] = useState(false)
  const [error, setError]       = useState(null)

  const validate = (file) => {
    if (!file) return 'No file selected.'
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXTS.includes(ext)) {
      return `Unsupported format. Use: ${ALLOWED_EXTS.join(', ')}`
    }
    if (file.size > 150 * 1024 * 1024) return 'File too large. Max 150 MB.'
    return null
  }

  const handleFile = useCallback((file) => {
    const err = validate(file)
    if (err) { setError(err); return }
    setError(null)
    onFile(file)
  }, [onFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onInput = (e) => {
    const file = e.target.files[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <label
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          minHeight: 200, padding: 32, borderRadius: 'var(--radius-lg)',
          border: `2px dashed ${dragOver ? 'var(--accent-green)' : 'rgba(255,255,255,0.12)'}`,
          background: dragOver ? 'rgba(0,255,136,0.05)' : 'rgba(255,255,255,0.02)',
          cursor: disabled ? 'not-allowed' : 'pointer',
          transition: 'all 0.25s ease',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {/* Icon */}
        <div style={{
          width: 64, height: 64, borderRadius: '50%',
          background: 'linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.15))',
          border: '1px solid rgba(0,255,136,0.25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 28, marginBottom: 16,
          transition: 'transform 0.2s',
          transform: dragOver ? 'scale(1.1)' : 'scale(1)',
        }}>📹</div>

        <p style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)', marginBottom: 6 }}>
          {dragOver ? 'Drop it here!' : 'Drag & drop a video'}
        </p>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: 16 }}>
          or click to browse • MP4, AVI, MOV, MKV, WebM • max 150 MB
        </p>

        <span className="btn btn-secondary" style={{ pointerEvents: 'none', fontSize: '0.85rem', padding: '8px 20px' }}>
          Browse File
        </span>

        <input
          type="file"
          accept="video/*"
          onChange={onInput}
          disabled={disabled}
          style={{ position: 'absolute', opacity: 0, width: 0, height: 0 }}
        />
      </label>

      {error && (
        <div style={{
          marginTop: 12, padding: '10px 16px', borderRadius: 'var(--radius-sm)',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          color: '#f87171', fontSize: '0.875rem',
        }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  )
}
