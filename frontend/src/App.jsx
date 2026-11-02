import { useState } from 'react'

const navItems = [
  { id: 'overview', label: 'Overview', icon: '◫' },
  { id: 'email', label: 'Email Analysis', icon: '✉' },
  { id: 'url', label: 'URL Analysis', icon: '↗' },
  { id: 'history', label: 'Scan History', icon: '◷' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
]

const scans = [
  { target: 'Invoice update required', detail: 'Email · 2 min ago', status: 'Threat', tone: 'threat', icon: '✉' },
  { target: 'https://portal.acme-example.com', detail: 'URL · 18 min ago', status: 'Clean', tone: 'clean', icon: '↗' },
  { target: 'Shared document notice', detail: 'Email · 42 min ago', status: 'Review', tone: 'review', icon: '✉' },
  { target: 'https://help.example.org', detail: 'URL · 1 hr ago', status: 'Clean', tone: 'clean', icon: '↗' },
]

function DemoNotice() {
  return (
    <div className="demo-notice" role="note">
      <span className="notice-icon">i</span>
      <span><strong>Demo data</strong> · Overview metrics, activity charts, and records labeled DEMO DATA are sample records, not live API results.</span>
    </div>
  )
}

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
const API_BASE_URL = (configuredApiBaseUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '')

function formatProbability(value) {
  return `${(value * 100).toFixed(1)}%`
}

function isValidPrediction(value) {
  return value &&
    (value.prediction === 0 || value.prediction === 1) &&
    (value.label === 'PHISHING' || value.label === 'LEGITIMATE') &&
    Number.isFinite(value.legitimate_probability) &&
    value.legitimate_probability >= 0 && value.legitimate_probability <= 1 &&
    Number.isFinite(value.phishing_probability) &&
    value.phishing_probability >= 0 && value.phishing_probability <= 1
}

function requestErrorMessage(status, kind) {
  if (status === 422) {
    return kind === 'email'
      ? 'The email text could not be analyzed. Check that it contains usable content and try again.'
      : 'The URL is not valid. Check its format and try again.'
  }
  if (status === 503) return 'The analysis model is temporarily unavailable. Please try again later.'
  return 'The analysis request could not be completed. Please try again.'
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  )
}

function StatusPill({ tone, children }) {
  return <span className={`status-pill ${tone}`}><span className="status-dot" />{children}</span>
}

function ActivityChart() {
  const bars = [34, 52, 43, 72, 56, 86, 64, 48, 76, 58, 92, 68, 45, 74, 55, 83, 62, 96, 70, 52, 78, 61, 88, 66]
  return (
    <div className="chart-wrap" aria-label="Illustrative hourly scan activity, demo data">
      <div className="chart-bars">
        {bars.map((height, index) => <span key={index} style={{ height: `${height}%` }} className={index > 16 ? 'bar muted-bar' : 'bar'} />)}
      </div>
      <div className="chart-axis"><span>12 AM</span><span>6 AM</span><span>12 PM</span><span>6 PM</span><span>Now</span></div>
    </div>
  )
}

function Overview({ onNavigate }) {
  return (
    <>
      <SectionHeading eyebrow="Security overview" title="Good afternoon, Analyst" description="A clear view of your demo security activity." action={<button className="button button-primary" onClick={() => onNavigate('email')}><span>＋</span> New analysis</button>} />
      <DemoNotice />
      <div className="metric-grid">
        <article className="metric-card"><div className="metric-top"><span>Total scans</span><span className="metric-icon cyan-icon">⌁</span></div><div className="metric-value">1,284</div><div className="metric-foot"><span className="trend">↗ 12.8%</span><span>vs. previous period</span></div><div className="metric-spark spark-cyan">▁▂▃▂▄▃▅▄▆▅▇</div></article>
        <article className="metric-card"><div className="metric-top"><span>Threats detected</span><span className="metric-icon red-icon">⌑</span></div><div className="metric-value">38</div><div className="metric-foot"><span className="trend trend-red">↗ 4.2%</span><span>vs. previous period</span></div><div className="metric-spark spark-red">▁▂▁▃▂▄▂▅▄▆▅</div></article>
        <article className="metric-card"><div className="metric-top"><span>Clean results</span><span className="metric-icon green-icon">✓</span></div><div className="metric-value">1,246</div><div className="metric-foot"><span className="trend">97.0%</span><span>of demo scans</span></div><div className="metric-spark spark-green">▂▃▃▄▅▄▆▅▆▇▆</div></article>
      </div>
      <div className="dashboard-grid">
        <section className="panel activity-panel">
          <div className="panel-heading"><div><h2>Scan activity</h2><p>Illustrative activity over the last 24 hours</p></div><span className="range-select">Last 24 hours <span>⌄</span></span></div>
          <ActivityChart />
          <div className="chart-legend"><span><i className="legend-swatch" /> Demo scans</span><span className="legend-caption">Illustrative data</span></div>
        </section>
        <section className="panel posture-panel">
          <div className="panel-heading"><div><h2>Security posture</h2><p>Demo summary</p></div><span className="more-button">···</span></div>
          <div className="posture-score"><div className="score-ring"><div><strong>97</strong><span>/ 100</span></div></div><div><strong className="posture-label">All systems calm</strong><p>Sample status only</p></div></div>
          <div className="posture-divider" />
          <div className="posture-row"><span><i className="mini-dot dot-cyan" /> Email checks</span><strong>824</strong></div>
          <div className="posture-row"><span><i className="mini-dot dot-purple" /> URL checks</span><strong>460</strong></div>
        </section>
      </div>
      <RecentScans rows={scans.slice(0, 3)} onViewAll={() => onNavigate('history')} />
    </>
  )
}

function RecentScans({ rows, onViewAll }) {
  return (
    <section className="panel recent-panel">
      <div className="panel-heading"><div><h2>Recent scans <span className="demo-tag">DEMO DATA</span></h2><p>Sample records for interface preview</p></div><button className="text-button" onClick={onViewAll}>View all <span>→</span></button></div>
      <div className="table-scroll"><table><thead><tr><th>SCAN TARGET</th><th>TYPE</th><th>TIME</th><th>STATUS</th><th aria-label="Details" /></tr></thead><tbody>{rows.map((scan) => <tr key={scan.target}><td><div className="target-cell"><span className={`scan-icon ${scan.icon === '✉' ? 'mail-icon' : 'link-icon'}`}>{scan.icon}</span><span className="target-text">{scan.target}</span></div></td><td>{scan.detail.split(' · ')[0]}</td><td>{scan.detail.split(' · ')[1]}</td><td><StatusPill tone={scan.tone}>{scan.status}</StatusPill></td><td><button className="row-more" aria-label={`More about ${scan.target}`}>···</button></td></tr>)}</tbody></table></div>
    </section>
  )
}

function AnalysisPage({ kind, onPrediction }) {
  const isEmail = kind === 'email'
  const [value, setValue] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const title = isEmail ? 'Email Analysis' : 'URL Analysis'
  async function submit(event) {
    event.preventDefault()
    if (!value.trim() || isLoading) return

    setResult(null)
    setError('')
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/${kind}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(isEmail ? { text: value } : { url: value }),
      })

      if (!response.ok) {
        setError(requestErrorMessage(response.status, kind))
        return
      }

      let responseData
      try {
        responseData = await response.json()
      } catch {
        setError('The analysis service returned an unreadable response. Please try again.')
        return
      }

      if (!isValidPrediction(responseData)) {
        setError('The analysis service returned an unexpected result. Please try again.')
        return
      }
      setResult(responseData)
      onPrediction({
        scanType: isEmail ? 'Email' : 'URL',
        timestamp: new Date().toISOString(),
        label: responseData.label,
        legitimate_probability: responseData.legitimate_probability,
        phishing_probability: responseData.phishing_probability,
      })
    } catch {
      setError('Could not reach the analysis API. Check your connection and try again.')
    } finally {
      setIsLoading(false)
    }
  }
  return (
    <>
      <SectionHeading eyebrow="Analysis workspace" title={title} description={isEmail ? 'Analyze email text with the phishing detection model.' : 'Analyze a URL with the phishing detection model.'} />
      <div className="analysis-layout">
        <section className="panel form-panel">
          <div className="panel-heading"><div><h2>{isEmail ? 'Email content' : 'Website address'}</h2><p>{isEmail ? 'Paste email text for a model prediction.' : 'Enter an HTTP or HTTPS address for a model prediction.'}</p></div><span className="input-type-icon">{isEmail ? '✉' : '↗'}</span></div>
          <form onSubmit={submit} aria-busy={isLoading}>
            <label htmlFor="analysis-input">{isEmail ? 'Email text' : 'URL'}</label>
            {isEmail
              ? <textarea id="analysis-input" value={value} onChange={(event) => { setValue(event.target.value); setResult(null); setError('') }} placeholder="Paste email text here…" rows="9" disabled={isLoading} />
              : <input id="analysis-input" type="text" inputMode="url" value={value} onChange={(event) => { setValue(event.target.value); setResult(null); setError('') }} placeholder="https://example.com/page" disabled={isLoading} />}
            <div className="input-hint">{isEmail ? 'Text is sent to the analysis API and is not shown in the result.' : 'The API analyzes URL features only; it does not visit the submitted address.'}</div>
            <button className="button button-primary submit-button" type="submit" disabled={!value.trim() || isLoading}>
              {isLoading ? <><span className="loading-spinner" aria-hidden="true" /> Analyzing…</> : <>{isEmail ? 'Analyze email' : 'Analyze URL'} <span>→</span></>}
            </button>
          </form>
        </section>
        <section className="panel result-panel">
          <div className="panel-heading"><div><h2>Analysis result</h2><p>Model API</p></div><span className="live-tag">MODEL API</span></div>
          {isLoading
            ? <div className="empty-state loading-state" role="status" aria-live="polite"><span className="loading-spinner large-spinner" aria-hidden="true" /><h3>Analyzing {isEmail ? 'email' : 'URL'}…</h3><p>Waiting for the model response.</p></div>
            : error
              ? <div className="result-state error-state" role="alert"><div className="result-symbol">!</div><h3>Analysis unavailable</h3><p>{error}</p></div>
              : result
                ? <div className={`result-state prediction-result ${result.label === 'PHISHING' ? 'threat-result' : 'clean-result'}`} aria-live="polite">
                    <div className="result-symbol">{result.label === 'PHISHING' ? '!' : '✓'}</div>
                    <span className="result-kicker">REAL MODEL PREDICTION</span>
                    <h3>{result.label}</h3>
                    <p>Prediction returned by the email and URL analysis API.</p>
                    <div className="probability-list">
                      <div className="probability-row"><span>Phishing probability</span><strong>{formatProbability(result.phishing_probability)}</strong></div>
                      <div className="probability-row"><span>Legitimate probability</span><strong>{formatProbability(result.legitimate_probability)}</strong></div>
                    </div>
                  </div>
                : <div className="empty-state"><div className="empty-art"><span className="empty-ring ring-one" /><span className="empty-ring ring-two" /><span className="empty-symbol">{isEmail ? '✉' : '↗'}</span></div><h3>Ready to analyze</h3><p>Submit {isEmail ? 'email content' : 'a website address'} to receive a prediction from the model API.</p></div>}
        </section>
      </div>
    </>
  )
}

function RealScanHistory({ entries }) {
  return (
    <section className="panel recent-panel real-history-panel">
      <div className="panel-heading">
        <div><h2>Real model scans <span className="live-tag">API RESULTS</span></h2><p>Successful predictions from this page session; submitted content is not retained.</p></div>
      </div>
      {entries.length === 0
        ? <div className="empty-state"><h3>No real scans yet</h3><p>Successful email and URL predictions will appear here while this page remains open.</p></div>
        : <div className="table-scroll"><table><thead><tr><th>TYPE</th><th>TIME</th><th>PREDICTION</th><th>PHISHING</th><th>LEGITIMATE</th></tr></thead><tbody>
          {entries.map((entry, index) => (
            <tr key={`${entry.timestamp}-${index}`}>
              <td>{entry.scanType}</td>
              <td>{new Date(entry.timestamp).toLocaleString()}</td>
              <td><StatusPill tone={entry.label === 'PHISHING' ? 'threat' : 'clean'}>{entry.label}</StatusPill></td>
              <td>{formatProbability(entry.phishing_probability)}</td>
              <td>{formatProbability(entry.legitimate_probability)}</td>
            </tr>
          ))}
        </tbody></table></div>}
    </section>
  )
}

function HistoryPage({ entries }) {
  return (
    <>
      <SectionHeading eyebrow="Activity" title="Scan History" description="Real predictions from this session and separately labeled sample records." />
      <RealScanHistory entries={entries} />
      <DemoNotice />
      <RecentScans rows={scans} onViewAll={() => {}} />
    </>
  )
}

function SettingsPage() {
  return <><SectionHeading eyebrow="Workspace" title="Settings" description="A placeholder for future application preferences." /><DemoNotice /><section className="panel settings-panel"><div className="settings-icon">⚙</div><div><h2>Settings are coming later</h2><p>This frontend shell does not store preferences or control security features. Configuration options can be added when the application API and product requirements are defined.</p><span className="placeholder-chip">PLACEHOLDER</span></div></section></>
}

export default function App() {
  const [activePage, setActivePage] = useState('overview')
  const [scanHistory, setScanHistory] = useState([])
  const activeItem = navItems.find((item) => item.id === activePage)
  function recordSuccessfulScan(entry) {
    setScanHistory((currentHistory) => [entry, ...currentHistory])
  }
  function navigate(page) {
    setActivePage(page)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  return (
    <div className="app-shell">
        <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><span /></div><div className="brand-copy"><strong>Sentinel</strong><span>SECURITY CONSOLE</span></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">{navItems.map((item) => <button key={item.id} className={`nav-item ${activePage === item.id ? 'active' : ''}`} onClick={() => navigate(item.id)}><span className="nav-icon">{item.icon}</span><span>{item.label}</span>{item.id === 'history' && <span className="nav-count">4</span>}</button>)}</nav>
        <div className="sidebar-bottom"><div className="connection-card"><span className="connection-indicator" /><div><strong>Analysis API</strong><span>Used for email and URL scans</span></div></div><div className="profile"><div className="avatar">A</div><div className="profile-copy"><strong>Analyst</strong><span>Student project</span></div><button aria-label="Profile menu" className="profile-menu">···</button></div></div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div className="breadcrumb"><span>Console</span><span className="crumb-separator">/</span><strong>{activeItem?.label}</strong></div><div className="topbar-right"><span className="environment-badge"><i />{activePage === 'email' || activePage === 'url' ? 'MODEL API' : 'DEMO MODE'}</span><span className="topbar-divider" /><button className="icon-button" aria-label="Notifications">♧<i className="notification-dot" /></button><div className="top-avatar">A</div></div></header>
        <div className="page-content">
          {activePage === 'overview' && <Overview onNavigate={navigate} />}
          {(activePage === 'email' || activePage === 'url') && <AnalysisPage key={activePage} kind={activePage} onPrediction={recordSuccessfulScan} />}
          {activePage === 'history' && <HistoryPage entries={scanHistory} />}
          {activePage === 'settings' && <SettingsPage />}
          <footer className="page-footer"><span>Sentinel Security Console <span className="footer-dot">·</span> Frontend preview</span><span>DEMO DATA ONLY</span></footer>
        </div>
      </main>
    </div>
  )
}
