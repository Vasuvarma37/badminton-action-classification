import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer style={{
      position: 'relative', zIndex: 1,
      borderTop: '1px solid rgba(255,255,255,0.06)',
      background: 'rgba(8,11,20,0.95)',
      backdropFilter: 'blur(20px)',
      padding: '48px 0 24px',
    }}>
      <div className="container">
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 40,
          marginBottom: 40,
        }}>
          {/* Brand */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: 'linear-gradient(135deg, #00ff88, #00d4ff)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 16,
              }}>🏸</div>
              <span style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 800, fontSize: '1rem', color: '#f0f4ff' }}>
                Badminton<span className="gradient-text">AI</span>
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.7 }}>
              AI-powered badminton shot classification using BiGRU + Attention neural networks.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <h4 style={{ fontFamily: 'Outfit, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 14 }}>Navigation</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[['/', 'Home'], ['/classify', 'Classify Shot'], ['/about', 'About Model']].map(([to, label]) => (
                <Link key={to} to={to} style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.875rem', transition: 'color 0.2s' }}
                  onMouseEnter={e => e.target.style.color = '#00ff88'}
                  onMouseLeave={e => e.target.style.color = 'var(--text-muted)'}
                >
                  {label}
                </Link>
              ))}
            </div>
          </div>

          {/* Technology */}
          <div>
            <h4 style={{ fontFamily: 'Outfit, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 14 }}>Technology Stack</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {['BiGRU + Attention (PyTorch)', 'MediaPipe Pose (33 landmarks)', 'FastAPI (Python backend)', 'React + Vite (Frontend)', 'Docker + Render (Deployment)'].map(t => (
                <span key={t} style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>• {t}</span>
              ))}
            </div>
          </div>

          {/* Model Info */}
          <div>
            <h4 style={{ fontFamily: 'Outfit, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 14 }}>Model Details</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                ['Architecture', 'Bidirectional GRU + Additive Attention'],
                ['Input', '30 frames × 26 features (13 joints)'],
                ['Classes', '4 badminton shot types'],
                ['Framework', 'PyTorch 2.0+'],
              ].map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem', display: 'block' }}>{k}</span>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.83rem' }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexWrap: 'wrap', gap: 12,
          paddingTop: 20,
          borderTop: '1px solid rgba(255,255,255,0.06)',
        }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            © 2025 BadmintonAI — Built with PyTorch, MediaPipe & React. Deployed on Render.
          </p>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <span className="badge badge-green">Production Ready</span>
            <span className="badge badge-cyan">v1.0.0</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
