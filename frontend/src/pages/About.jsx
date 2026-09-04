const ARCHITECTURE = [
  { layer: 'Input',         shape: '(B, 30, 26)',   desc: '30 frames, 13 joints × 2 coords' },
  { layer: 'LayerNorm',     shape: '(B, 30, 26)',   desc: 'Stabilize input across feature dim' },
  { layer: 'BiGRU × 3',    shape: '(B, 30, 512)',  desc: 'Hidden=256 per direction, stacked 3 layers' },
  { layer: 'Attention',     shape: '(B, 512)',      desc: 'Bahdanau-style additive attention over time' },
  { layer: 'FC + GELU',    shape: '(B, 128)',      desc: 'Intermediate fully-connected layer' },
  { layer: 'Dropout(0.4)', shape: '(B, 128)',      desc: 'Regularization' },
  { layer: 'FC (Output)',  shape: '(B, 4)',        desc: 'Logits for 4 action classes' },
]

const JOINTS = [
  'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow',
  'L-Wrist',    'R-Wrist',    'L-Hip',   'R-Hip',
  'L-Knee',     'R-Knee',     'L-Ankle', 'R-Ankle',
  'Nose',
]

export default function About() {
  return (
    <div className="section" style={{ paddingTop: 100 }}>
      <div className="container" style={{ maxWidth: 900 }}>

        <div style={{ marginBottom: 48 }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.8rem)', marginBottom: 12 }}>
            About the <span className="gradient-text">Model</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 600, fontSize: '1.05rem' }}>
            A deep-dive into the architecture, training setup, and design decisions behind the BiGRU + Attention classifier.
          </p>
        </div>

        {/* Overview */}
        <div className="card" style={{ padding: 32, marginBottom: 24 }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: 16 }}>Model Overview</h2>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: 16 }}>
            The classifier uses a <strong style={{ color: '#f0f4ff' }}>3-layer Bidirectional GRU</strong> with{' '}
            <strong style={{ color: '#f0f4ff' }}>Bahdanau-style additive attention</strong> to classify badminton shots from skeleton keypoint sequences.
          </p>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            Input videos are processed by <strong style={{ color: '#f0f4ff' }}>MediaPipe Pose</strong>, which extracts
            33 body landmarks per frame. These are mapped to a canonical 13-joint skeleton, giving 26 features per frame
            (x, y per joint). 30 uniformly-spaced frames are sampled per video.
          </p>
        </div>

        {/* Architecture table */}
        <div className="card" style={{ padding: 32, marginBottom: 24 }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: 20 }}>Network Architecture</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  {['Layer', 'Output Shape', 'Description'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '10px 16px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ARCHITECTURE.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '12px 16px', color: 'var(--accent-cyan)', fontFamily: 'monospace', fontWeight: 600 }}>
                      {row.layer}
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--accent-green)', fontFamily: 'monospace' }}>
                      {row.shape}
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>
                      {row.desc}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Why BiGRU */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 20, marginBottom: 24 }}>
          {[
            {
              title: 'Why BiGRU?', icon: '↔️', color: '#00ff88',
              text: 'A forward-only RNN sees only past frames. BiGRU also runs in reverse, giving each timestep a richer hidden state that captures the full stroke arc — crucial for distinguishing similar shots that differ in their follow-through.',
            },
            {
              title: 'Why 30 Frames?', icon: '🎞️', color: '#00d4ff',
              text: '10 frames captures only a slice of the stroke. 30 frames spans preparation → impact → recovery, giving the model enough temporal context to distinguish forehand clear from forehand drive.',
            },
            {
              title: 'Why Attention?', icon: '👁️', color: '#a78bfa',
              text: 'Not all 30 frames are equally informative — the impact frame matters more than recovery. Additive attention learns a soft weighting over time steps, focusing the classifier on the most discriminative moments.',
            },
          ].map(c => (
            <div key={c.title} className="card" style={{ padding: 26 }}>
              <div style={{ fontSize: 28, marginBottom: 12 }}>{c.icon}</div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: c.color, marginBottom: 10 }}>{c.title}</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', lineHeight: 1.7 }}>{c.text}</p>
            </div>
          ))}
        </div>

        {/* Joint mapping */}
        <div className="card" style={{ padding: 32, marginBottom: 24 }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: 16 }}>13-Joint Skeleton</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 20, fontSize: '0.9rem' }}>
            MediaPipe's 33 landmarks are projected onto a canonical 13-joint skeleton shared by both the MediaPipe and BST datasets.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {JOINTS.map((j, i) => (
              <div key={j} style={{
                padding: '6px 14px', borderRadius: 8,
                background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)',
                fontSize: '0.82rem', color: 'var(--text-secondary)',
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{ color: 'var(--accent-green)', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {String(i).padStart(2, '0')}
                </span>
                {j}
              </div>
            ))}
          </div>
        </div>

        {/* Training config */}
        <div className="card" style={{ padding: 32 }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: 20 }}>Training Configuration</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            {[
              ['Batch Size',    '64'],
              ['Max Epochs',    '500 (early stop @ 80)'],
              ['Optimizer',     'Adam (lr=1e-3)'],
              ['LR Schedule',   'ReduceLROnPlateau (×0.5)'],
              ['Weight Decay',  '1e-4 (L2 regularization)'],
              ['Dropout',       '0.4'],
              ['Hidden Units',  '256 per direction'],
              ['GRU Layers',    '3 stacked'],
            ].map(([k, v]) => (
              <div key={k} style={{
                padding: '14px 16px', borderRadius: 10,
                background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)',
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{k}</div>
                <div style={{ color: 'var(--accent-cyan)', fontFamily: 'monospace', fontWeight: 600, fontSize: '0.9rem' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
