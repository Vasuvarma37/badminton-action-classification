import { useState, useRef, useCallback, useEffect } from 'react'
import VideoUploader from '../components/VideoUploader'
import SkeletonViz from '../components/SkeletonViz'
import ConfidenceChart from '../components/ConfidenceChart'
import AttentionViz from '../components/AttentionViz'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const CLASS_META = {
  backhand_drive:    { emoji: '🔙', label: 'Backhand Drive' },
  backhand_net_shot: { emoji: '🎯', label: 'Backhand Net Shot' },
  forehand_clear:    { emoji: '💫', label: 'Forehand Clear' },
  forehand_drive:    { emoji: '⚡', label: 'Forehand Drive' },
}

export default function Classify() {
  const [file,        setFile]        = useState(null)
  const [videoUrl,    setVideoUrl]    = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [result,      setResult]      = useState(null)
  const [error,       setError]       = useState(null)
  const [activeFrame, setActiveFrame] = useState(null)

  const videoRef = useRef(null)

  // When user picks a file, create a local object URL for the video preview
  const handleFile = useCallback((f) => {
    setFile(f)
    setResult(null)
    setError(null)
    setActiveFrame(null)
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoUrl(URL.createObjectURL(f))
  }, [videoUrl])

  // Clean up on unmount
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])

  // ── Skeleton frame driven by video currentTime ──
  const [skeletonFrame, setSkeletonFrame] = useState(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video || !result?.keypoints?.length) return

    const totalFrames = result.keypoints.length
    const onTimeUpdate = () => {
      const pct = video.currentTime / (video.duration || 1)
      const idx = Math.min(totalFrames - 1, Math.floor(pct * totalFrames))
      setSkeletonFrame(idx)
    }
    video.addEventListener('timeupdate', onTimeUpdate)
    return () => video.removeEventListener('timeupdate', onTimeUpdate)
  }, [result])

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('video', file)

    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form })
      const data = await res.json()

      if (!res.ok) {
        setError(data.detail || `Server error ${res.status}`)
      } else if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (e) {
      setError('Network error — is the API server running?')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setActiveFrame(null)
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoUrl(null)
  }

  const meta = result ? (CLASS_META[result.class] || { emoji: '🏸', label: result.class }) : null

  return (
    <div className="section" style={{ paddingTop: 100 }}>
      <div className="container" style={{ maxWidth: 1100 }}>

        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.8rem)', marginBottom: 10 }}>
            Shot <span className="gradient-text">Classifier</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Upload a video clip of a badminton player to classify the shot type.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: result ? '1fr 1fr' : '1fr',
          gap: 28,
          transition: 'all 0.4s ease',
        }}>

          {/* Left Column — Upload + Video */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Uploader */}
            {!file && (
              <div className="card" style={{ padding: 28 }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16, color: 'var(--text-secondary)' }}>
                  STEP 1 — Upload Video
                </h3>
                <VideoUploader onFile={handleFile} disabled={loading} />
              </div>
            )}

            {/* Video Player with skeleton overlay */}
            {videoUrl && (
              <div className="card" style={{ padding: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                    📹 {file?.name}
                  </span>
                  <button onClick={reset} className="btn btn-ghost" style={{ fontSize: '0.8rem', padding: '4px 12px' }}>
                    ✕ Remove
                  </button>
                </div>

                {/* Video + skeleton overlay container */}
                <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: '#000' }}>
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    style={{ width: '100%', display: 'block', maxHeight: 400 }}
                    onLoadedMetadata={() => setSkeletonFrame(0)}
                  />
                  {/* Skeleton overlay — responsive via ResizeObserver in SkeletonViz */}
                  {result?.keypoints?.length > 0 && (
                    <SkeletonViz
                      keypoints={result.keypoints}
                      frameIndex={activeFrame !== null ? activeFrame : skeletonFrame}
                      videoRef={videoRef}
                    />
                  )}
                </div>

                {/* Action button */}
                {!result && (
                  <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: 16, justifyContent: 'center', fontSize: '0.95rem', padding: '12px' }}
                  >
                    {loading
                      ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Analyzing...</>
                      : '🧠 Classify Shot'}
                  </button>
                )}
              </div>
            )}

            {/* Error */}
            {error && (
              <div style={{
                padding: '16px 20px', borderRadius: 'var(--radius-md)',
                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                color: '#f87171',
              }}>
                <strong>⚠️ Error:</strong> {error}
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="card" style={{ padding: 32, textAlign: 'center' }}>
                <div className="spinner" style={{ margin: '0 auto 16px' }} />
                <p style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Extracting keypoints & running inference...</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: 6 }}>
                  MediaPipe is processing 30 frames. This may take 10–30 seconds.
                </p>
              </div>
            )}
          </div>

          {/* Right Column — Results */}
          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }} className="animate-fade-up">

              {/* Prediction badge */}
              <div className="card" style={{
                padding: 28,
                background: 'linear-gradient(135deg, rgba(0,255,136,0.06), rgba(0,212,255,0.06))',
                border: '1px solid rgba(0,255,136,0.2)',
                boxShadow: 'var(--shadow-card), var(--shadow-glow)',
              }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
                  Predicted Shot
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 }}>
                  <span style={{ fontSize: 48 }}>{meta.emoji}</span>
                  <div>
                    <h2 style={{ fontFamily: 'Outfit', fontSize: '1.6rem', fontWeight: 800, color: '#00ff88', lineHeight: 1.2 }}>
                      {meta.label}
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
                      Confidence: <strong style={{ color: '#f0f4ff' }}>{(result.confidence * 100).toFixed(1)}%</strong>
                    </p>
                  </div>
                </div>

                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${(result.confidence * 100).toFixed(1)}%` }} />
                </div>
              </div>

              {/* All class scores */}
              <div className="card" style={{ padding: 24 }}>
                <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 18 }}>
                  Class Probabilities
                </h3>
                <ConfidenceChart
                  classes={result.classes}
                  scores={result.scores}
                  predicted={result.class}
                />
              </div>

              {/* Attention visualization */}
              {result.attention?.length > 0 && (
                <div className="card" style={{ padding: 24 }}>
                  <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 14 }}>
                    Frame Attention Weights
                  </h3>
                  <AttentionViz
                    attention={result.attention}
                    activeFrame={activeFrame}
                    onFrameHover={setActiveFrame}
                  />
                </div>
              )}

              {/* Re-classify / Reset */}
              <div style={{ display: 'flex', gap: 12 }}>
                <button onClick={reset} className="btn btn-secondary" style={{ flex: 1, justifyContent: 'center' }}>
                  🔄 New Video
                </button>
                <button onClick={handleSubmit} disabled={loading} className="btn btn-ghost" style={{ flex: 1, justifyContent: 'center' }}>
                  Re-run
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
