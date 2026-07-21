import { useState } from 'react'
import { Bot, Send, Sparkles } from 'lucide-react'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { tenApi } from '../services/api'
import type { ChatTurn, ExplainResponse } from '../types'

interface DisplayMessage {
  role: 'user' | 'assistant'
  content: string
  response?: ExplainResponse
}

const SUGGESTIONS = [
  'Why was the last signal rejected?',
  'Explain the current market.',
  'What needs to change before TEN can publish a signal?',
  'Which engine has the greatest influence right now?',
]

function AssistantAnswer({ response }: { response: ExplainResponse }) {
  const { explanation, error, explainability_score: score, evidence } = response
  if (!explanation) {
    return <div className="explain-panel assistant__answer"><p className="explain-panel__error">{error ?? 'No explanation could be generated.'}</p></div>
  }
  return (
    <div className="explain-panel assistant__answer">
      <p className="explain-panel__summary">{explanation.summary}</p>
      {explanation.primary_reasons.length > 0 && (
        <div className="explain-block"><p className="explain-columns__label">Supporting</p><ul>{explanation.primary_reasons.map((item, index) => <li key={index}>{item}</li>)}</ul></div>
      )}
      {explanation.opposing_factors.length > 0 && (
        <div className="explain-block"><p className="explain-columns__label">Opposing</p><ul>{explanation.opposing_factors.map((item, index) => <li key={index}>{item}</li>)}</ul></div>
      )}
      {explanation.required_for_change.length > 0 && (
        <div className="explain-block"><p className="explain-columns__label">What would need to change</p><ul>{explanation.required_for_change.map((item, index) => <li key={index}>{item}</li>)}</ul></div>
      )}
      {evidence.length > 0 && (
        <div className="explain-evidence"><span>Evidence</span>{evidence.map((item) => <code key={`${item.source}-${item.reference_id}`} title={item.timestamp ?? ''}>{item.source} #{item.reference_id.slice(0, 8)}</code>)}</div>
      )}
      <p className="explain-score"><span>{score.percent}% explanation confidence · {score.engines_available}/{score.engines_total} engines available</span></p>
    </div>
  )
}

export function AssistantPage() {
  const { selection } = useActiveSelection()
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function send(message: string) {
    const text = message.trim()
    if (!text || sending) return
    setInput('')
    setError(null)
    // Conversation history travels inside the same grounded request every other /explain/* route
    // uses — an assistant answer can never disagree with what the dashboard panels show.
    const history: ChatTurn[] = messages.map(({ role, content }) => ({ role, content }))
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setSending(true)
    try {
      const response = await tenApi.explainChat(text, history, selection.instrument, selection.timeframe)
      const content = response.explanation?.summary ?? response.error ?? 'No explanation could be generated.'
      setMessages((prev) => [...prev, { role: 'assistant', content, response }])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'chat request failed')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">EXPLAINABILITY</p><h1>AI <em>assistant.</em></h1></div>
        <div className="page-icon"><Sparkles size={25} /></div>
      </header>
      {error && <div className="alert"><span>Request failed</span>{error}</div>}
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">{selection.instrument} · {selection.timeframe}</p><h2>Ask about TEN's current analysis</h2></div><span>grounded in TEN only</span></div>
        <div className="panel-body assistant">
          <div className="assistant__messages">
            {messages.length === 0 && (
              <div className="assistant__empty">
                <Bot size={26} />
                <p>Ask anything about TEN's current pipeline state, a rejected scenario, or a past decision. Every answer is grounded in TEN's own engine outputs, with cited evidence — nothing invented, nothing external.</p>
                <div className="assistant__suggestions">
                  {SUGGESTIONS.map((suggestion) => <button key={suggestion} onClick={() => void send(suggestion)}>{suggestion}</button>)}
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <div className={`assistant__message assistant__message--${message.role}`} key={index}>
                {message.role === 'assistant' && <p className="assistant__bubble-label"><Bot size={12} /> TEN Assistant</p>}
                {message.role === 'user' ? message.content : message.response ? <AssistantAnswer response={message.response} /> : <p>{message.content}</p>}
              </div>
            ))}
            {sending && (
              <div className="assistant__message assistant__message--assistant">
                <p className="assistant__bubble-label"><Bot size={12} /> TEN Assistant</p>
                <div className="explain-panel assistant__answer explain-panel--loading">
                  <div className="skeleton skeleton-line" style={{ width: '70%' }} />
                  <div className="skeleton skeleton-line" style={{ width: '50%' }} />
                </div>
              </div>
            )}
          </div>
          <form className="assistant__form" onSubmit={(event) => { event.preventDefault(); void send(input) }}>
            <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Why was confidence low?" disabled={sending} />
            <button type="submit" disabled={sending || !input.trim()}><Send size={14} />Send</button>
          </form>
        </div>
      </section>
    </div>
  )
}
