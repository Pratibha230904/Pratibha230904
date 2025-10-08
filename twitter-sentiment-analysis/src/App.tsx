import { useMemo, useState } from 'react'
import Sentiment from 'sentiment'

const sentiment = new Sentiment()

type SentimentClass = 'positive' | 'negative' | 'neutral'

function clamp(min: number, value: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

export default function App() {
  const [text, setText] = useState<string>(
    'Just tried the new app update — absolutely loving the performance improvements! 🚀'
  )
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)

  const result = useMemo(() => {
    return sentiment.analyze(text)
  }, [text])

  const score = result.score
  const comparative = result.comparative
  const positiveWords = result.positive
  const negativeWords = result.negative

  const scoreClass: SentimentClass = score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral'

  const scorePercent = useMemo(() => {
    const normalized = (score + 10) / 20 // map approx [-10, 10] -> [0,1]
    return clamp(0, normalized, 1) * 100
  }, [score])

  async function handleAnalyze() {
    setIsAnalyzing(true)
    // Simulate a brief async step to keep UI responsive
    await new Promise((r) => setTimeout(r, 120))
    setIsAnalyzing(false)
  }

  return (
    <div className="app">
      <div className="card">
        <div className="header">
          <svg className="logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
            <path d="M22 5.84c-.65.29-1.35.48-2.07.57a3.63 3.63 0 0 0 1.6-2 7.27 7.27 0 0 1-2.3.88A3.63 3.63 0 0 0 12.1 8.2a10.3 10.3 0 0 1-7.48-3.79 3.63 3.63 0 0 0 1.12 4.84c-.56-.02-1.09-.17-1.56-.42v.04c0 1.76 1.25 3.22 2.9 3.55-.3.08-.63.12-.96.12-.23 0-.46-.02-.68-.06.46 1.46 1.8 2.52 3.38 2.55A7.3 7.3 0 0 1 3 17.82a10.29 10.29 0 0 0 5.58 1.64c6.69 0 10.35-5.55 10.35-10.36 0-.16 0-.32-.01-.48A7.4 7.4 0 0 0 22 5.84Z" fill="#6ea8fe"/>
          </svg>
          <div>
            <div className="title">Twitter Sentiment Analysis</div>
            <div className="subtitle">Analyze tweet sentiment instantly. Offline, private, and fast.</div>
          </div>
        </div>

        <div className="row">
          <textarea
            className="textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a tweet or type your text here..."
          />
          <div className="controls">
            <button className="btn" onClick={handleAnalyze} disabled={isAnalyzing}>
              {isAnalyzing ? 'Analyzing…' : 'Analyze Sentiment'}
            </button>
            <div className="meta">
              <div className="metric">
                <div className="label">Score</div>
                <div className="value">{score}</div>
              </div>
              <div className="metric">
                <div className="label">Comparative</div>
                <div className="value">{comparative.toFixed(3)}</div>
              </div>
              <div className="metric">
                <div className="label">Classification</div>
                <div className="value" style={{ color: scoreClass === 'positive' ? 'var(--positive)' : scoreClass === 'negative' ? 'var(--negative)' : 'var(--neutral)' }}>
                  {scoreClass.toUpperCase()}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="score" aria-label="Sentiment score visualization">
          <div
            className={`scoreFill ${scoreClass}`}
            style={{ width: `${scorePercent}%` }}
          />
        </div>
        <div className="legend">
          <span><i className="dot negative"/> Negative</span>
          <span><i className="dot neutral"/> Neutral</span>
          <span><i className="dot positive"/> Positive</span>
        </div>

        <div className="row" style={{ marginTop: 16 }}>
          <div className="metric" style={{ flex: 1 }}>
            <div className="label">Positive tokens</div>
            <div style={{ marginTop: 6 }}>
              {positiveWords.length ? positiveWords.map((w, i) => (
                <span key={w + i} style={{ color: 'var(--positive)', marginRight: 8 }}>#{w}</span>
              )) : <span style={{ color: 'var(--muted)' }}>None</span>}
            </div>
          </div>
          <div className="metric" style={{ flex: 1 }}>
            <div className="label">Negative tokens</div>
            <div style={{ marginTop: 6 }}>
              {negativeWords.length ? negativeWords.map((w, i) => (
                <span key={w + i} style={{ color: 'var(--negative)', marginRight: 8 }}>#{w}</span>
              )) : <span style={{ color: 'var(--muted)' }}>None</span>}
            </div>
          </div>
        </div>

        <div className="footer">
          Uses the open-source "sentiment" library. No data leaves your browser.
        </div>
      </div>
    </div>
  )
}

