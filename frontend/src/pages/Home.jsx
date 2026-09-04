import { Link } from 'react-router-dom'

const STEPS = [
  { icon: '📹', title: 'Upload Video', desc: 'Drop any badminton video clip — MP4, AVI, MOV, MKV supported up to 150 MB.' },
  { icon: '🦴', title: 'Skeleton Extraction', desc: 'MediaPipe extracts 33 pose landmarks from 30 uniformly sampled frames.' },
  { icon: '🧠', title: 'AI Classification', desc: 'BiGRU + Attention neural net classifies the shot type in under a second.' },
]

const CLASSES = [
  { name: 'Backhand Drive',    emoji: '🔙', color: '#00ff88', desc: 'Fast flat stroke hit from the backhand side' },
  { name: 'Backhand Net Shot', emoji: '🎯', color: '#00d4ff', desc: 'Delicate touch shot near the net on backhand' },
  { name: 'Forehand Clear',    emoji: '💫', color: '#a78bfa', desc: 'High defensive or attacking overhead clear' },
  { name: 'Forehand Drive',    emoji: '⚡', color: '#f59e0b', desc: 'Aggressive flat drive hit with the forehand' },
]

const STATS = [
  { value: '4',   label: 'Shot Classes' },
  { value: '30',  label: 'Frames Analyzed' },
  { value: '13',  label: 'Skeleton Joints' },
  { value: 'BiGRU', label: 'Model Type' },
]

export default function Home() {
  return (
    <div>
      {/* ── Hero ── */}
      <section style={{
        minHeight: '100vh', display: 'flex', alignItems: 'center',
        padding: '100px 0 60px', position: 'relative', overflow: 'hidden',
      }}>
        {/* Animated background orbs */}
        <div style={{
          position: 'absolute', top: '15%', right: '10%',
          width: 500, height: 500, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none',
          animation: 'floatY 6s ease-in-out infinite',
        }} />
        <div style={{
          position: 'absolute', bottom: '10%', left: '5%',
          width: 400, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none',
          animation: 'floatY 8s ease-in-out infinite reverse',
        }} />

        <div className="container" style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ maxWidth: 760 }}>
            <div className="badge badge-green animate-fade-up" style={{ marginBottom: 20 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor', display: 'inline-block' }} />
              AI-Powered · Real-time · Production Ready
            </div>

            <h1 className="animate-fade-up" style={{
              fontSize: 'clamp(2.4rem, 6vw, 4.2rem)',
              fontWeight: 900,
              lineHeight: 1.1,
              marginBottom: 20,
              animationDelay: '0.1s',
            }}>
              Classify Badminton Shots<br />
              <span className="gradient-text">with Neural Networks</span>
            </h1>

            <p className="animate-fade-up" style={{
              fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
              color: 'var(--text-secondary)',
              maxWidth: 560,
              marginBottom: 36,
              animationDelay: '0.2s',
            }}>
              Upload a video clip and our BiGRU + Attention model extracts skeleton keypoints from 30 frames to classify the shot in under a second.
            </p>

            <div className="animate-fade-up" style={{ display: 'flex', gap: 14, flexWrap: 'wrap', animationDelay: '0.3s' }}>
              <Link to="/classify" className="btn btn-primary" style={{ fontSize: '1rem', padding: '14px 28px' }}>
                🏸 Classify a Shot
              </Link>
              <Link to="/about" className="btn btn-secondary" style={{ fontSize: '1rem', padding: '14px 28px' }}>
                Learn More →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats ── */}
      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 20,
          }}>
            {STATS.map(s => (
              <div key={s.label} className="card" style={{ padding: '24px 20px', textAlign: 'center' }}>
                <div style={{ fontFamily: 'Outfit, sans-serif', fontSize: '2rem', fontWeight: 900 }} className="gradient-text">
                  {s.value}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 4 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="section">
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <h2 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.6rem)', marginBottom: 12 }}>
              How It <span className="gradient-text">Works</span>
            </h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: 480, margin: '0 auto' }}>
              Three steps from video to classification result.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 24 }}>
            {STEPS.map((step, i) => (
              <div key={step.title} className="card" style={{ padding: 32, position: 'relative' }}>
                <div style={{
                  position: 'absolute', top: 20, right: 20,
                  fontFamily: 'Outfit, sans-serif', fontWeight: 900,
                  fontSize: '3rem', color: 'rgba(255,255,255,0.04)',
                }}>
                  {i + 1}
                </div>
                <div style={{ fontSize: 36, marginBottom: 16 }}>{step.icon}</div>
                <h3 style={{ fontSize: '1.1rem', marginBottom: 10, color: 'var(--text-primary)' }}>
                  {step.title}
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.7 }}>
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Action Classes ── */}
      <section className="section" style={{ background: 'rgba(255,255,255,0.01)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <h2 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.6rem)', marginBottom: 12 }}>
              Recognized <span className="gradient-text">Shot Types</span>
            </h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: 480, margin: '0 auto' }}>
              Our model distinguishes four fundamental badminton shot categories.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 20 }}>
            {CLASSES.map(cls => (
              <div key={cls.name} className="card" style={{ padding: 28 }}>
                <div style={{
                  width: 52, height: 52, borderRadius: 14,
                  background: `${cls.color}18`,
                  border: `1px solid ${cls.color}35`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 24, marginBottom: 16,
                }}>
                  {cls.emoji}
                </div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: cls.color, marginBottom: 8 }}>
                  {cls.name}
                </h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', lineHeight: 1.6 }}>
                  {cls.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="section">
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="card" style={{
            maxWidth: 680, margin: '0 auto', padding: '60px 40px',
            background: 'linear-gradient(135deg, rgba(0,255,136,0.05) 0%, rgba(0,212,255,0.05) 100%)',
            border: '1px solid rgba(0,255,136,0.15)',
            boxShadow: 'var(--shadow-card), var(--shadow-glow)',
          }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🏸</div>
            <h2 style={{ fontSize: '2rem', marginBottom: 12 }}>Ready to Classify?</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 28 }}>
              Upload your badminton video and get instant AI-powered shot classification.
            </p>
            <Link to="/classify" className="btn btn-primary" style={{ fontSize: '1rem', padding: '14px 32px' }}>
              Start Classifying →
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
