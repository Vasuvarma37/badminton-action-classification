import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'

const NAV_LINKS = [
  { to: '/',         label: 'Home' },
  { to: '/classify', label: 'Classify' },
  { to: '/about',    label: 'About' },
]

export default function Navbar() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => setMenuOpen(false), [location])

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
      background: scrolled ? 'rgba(8,11,20,0.95)' : 'transparent',
      backdropFilter: scrolled ? 'blur(20px)' : 'none',
      borderBottom: scrolled ? '1px solid rgba(255,255,255,0.06)' : 'none',
      transition: 'all 0.3s ease',
    }}>
      <div className="container" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 64,
      }}>
        {/* Logo */}
        <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'linear-gradient(135deg, #00ff88, #00d4ff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18,
          }}>🏸</div>
          <span style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 800, fontSize: '1.1rem', color: '#f0f4ff' }}>
            Badminton<span className="gradient-text">AI</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }} className="desktop-nav">
          {NAV_LINKS.map(l => (
            <Link key={l.to} to={l.to} style={{
              padding: '8px 16px', borderRadius: 8, fontSize: '0.9rem', fontWeight: 500,
              textDecoration: 'none',
              color: location.pathname === l.to ? '#00ff88' : '#8899bb',
              background: location.pathname === l.to ? 'rgba(0,255,136,0.08)' : 'transparent',
              transition: 'all 0.2s',
            }}>
              {l.label}
            </Link>
          ))}
          <Link to="/classify" className="btn btn-primary" style={{ marginLeft: 12, padding: '8px 20px', fontSize: '0.88rem' }}>
            Try it Free
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button
          onClick={() => setMenuOpen(o => !o)}
          style={{ display: 'none', background: 'none', border: 'none', cursor: 'pointer', color: '#f0f4ff', fontSize: 22 }}
          className="hamburger"
          aria-label="Menu"
        >☰</button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div style={{
          position: 'absolute', top: 64, left: 0, right: 0,
          background: 'rgba(8,11,20,0.98)', backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          padding: '16px 24px 24px',
          display: 'flex', flexDirection: 'column', gap: 8,
        }}>
          {NAV_LINKS.map(l => (
            <Link key={l.to} to={l.to} style={{
              padding: '12px 16px', borderRadius: 8, fontSize: '1rem', fontWeight: 500,
              textDecoration: 'none',
              color: location.pathname === l.to ? '#00ff88' : '#8899bb',
              background: location.pathname === l.to ? 'rgba(0,255,136,0.08)' : 'transparent',
            }}>{l.label}</Link>
          ))}
          <Link to="/classify" className="btn btn-primary" style={{ marginTop: 8, textAlign: 'center', justifyContent: 'center' }}>
            Try it Free
          </Link>
        </div>
      )}

      <style>{`
        @media (max-width: 640px) {
          .desktop-nav { display: none !important; }
          .hamburger   { display: block !important; }
        }
      `}</style>
    </nav>
  )
}
