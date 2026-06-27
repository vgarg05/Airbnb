import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send, MapPin, Star, Wifi, Waves, Mountain, Building2,
  Home, Flame, Sparkles, ChevronRight, X, TreePine, Wind,
  Settings, Key, Lock, ShieldCheck, Eye, EyeOff
} from 'lucide-react'

// ─── Constants ───────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const STREAM_URL = `${API_BASE}/chat/stream`

const QUICK_CHIPS = [
  { label: 'Cozy cabin in mountains',  emoji: '🏔️' },
  { label: 'Family trip to Goa',        emoji: '🌴' },
  { label: 'Romantic Paris stay',       emoji: '🗼' },
  { label: 'Budget Tokyo studio',       emoji: '🗾' },
  { label: 'Overwater villa Maldives',  emoji: '🐠' },
  { label: 'Safari lodge Africa',       emoji: '🦁' },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function imgClass(type = '') {
  const map = { apartment: 'img-apartment', cabin: 'img-cabin', villa: 'img-villa', studio: 'img-studio' }
  return map[type.toLowerCase()] ?? 'img-apartment'
}

function typeClass(type = '') {
  const map = { apartment: 'type-apartment', cabin: 'type-cabin', villa: 'type-villa', studio: 'type-studio' }
  return map[type.toLowerCase()] ?? 'type-apartment'
}

function TypeIcon({ type, className = 'w-3.5 h-3.5' }) {
  const icons = { apartment: Building2, cabin: TreePine, villa: Home, studio: Wind }
  const Icon = icons[type?.toLowerCase()] ?? Building2
  return <Icon className={className} />
}

function AmenityIcon({ name }) {
  const n = name.toLowerCase()
  if (n.includes('wifi') || n.includes('internet'))  return <Wifi  className="w-3 h-3" />
  if (n.includes('pool') || n.includes('beach'))     return <Waves className="w-3 h-3" />
  if (n.includes('mountain') || n.includes('hike'))  return <Mountain className="w-3 h-3" />
  if (n.includes('fire') || n.includes('stove'))     return <Flame className="w-3 h-3" />
  return <Sparkles className="w-3 h-3" />
}

// ─── Markdown components ─────────────────────────────────────────────────────

const mdComponents = {
  h1: ({ children }) => <h1 className="text-lg font-bold text-neutral-900 mt-3 mb-1">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-bold text-neutral-900 mt-3 mb-1">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold text-neutral-800 mt-2 mb-0.5">{children}</h3>,
  p: ({ children }) => <p className="text-sm text-neutral-800 leading-relaxed mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="list-disc list-outside pl-4 mb-2 space-y-0.5 text-sm text-neutral-800">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-outside pl-4 mb-2 space-y-0.5 text-sm text-neutral-800">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-neutral-900">{children}</strong>,
  em: ({ children }) => <em className="italic text-neutral-700">{children}</em>,
  code: ({ inline, children }) =>
    inline
      ? <code className="bg-neutral-100 text-airbnb-600 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
      : <pre className="bg-neutral-100 rounded-xl p-3 overflow-x-auto text-xs font-mono text-neutral-700 my-2"><code>{children}</code></pre>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-airbnb-300 pl-3 italic text-neutral-600 my-2 text-sm">{children}</blockquote>
  ),
  hr: () => <hr className="border-neutral-200 my-3" />,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer"
       className="text-airbnb-500 underline underline-offset-2 hover:text-airbnb-600 transition-colors">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse border border-neutral-200 rounded-lg overflow-hidden">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="bg-neutral-100 font-semibold text-neutral-700 px-3 py-1.5 border border-neutral-200 text-left">{children}</th>,
  td: ({ children }) => <td className="px-3 py-1.5 border border-neutral-200 text-neutral-700">{children}</td>,
}

// ─── Blinking cursor ─────────────────────────────────────────────────────────
function StreamCursor() {
  return (
    <span className="inline-block w-0.5 h-4 bg-neutral-700 ml-0.5 align-middle animate-blink" />
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <AiAvatar />
      <div className="bubble-ai flex items-center gap-1.5 py-4 px-5">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  )
}

function AiAvatar() {
  return (
    <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm"
         style={{ background: 'linear-gradient(135deg,#FF385C,#E61E4D)' }}>
      <Sparkles className="w-4 h-4 text-white" />
    </div>
  )
}

function ListingCard({ listing }) {
  const topAmenities = listing.amenities.slice(0, 3)

  return (
    <div className="listing-card w-64 sm:w-72 flex-shrink-0 cursor-pointer group">
      <div className={`relative h-36 ${imgClass(listing.type)} overflow-hidden`}>
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-300" />

        <div className="absolute top-2.5 left-2.5">
          <span className={`type-badge ${typeClass(listing.type)} bg-white/90 backdrop-blur-sm`}>
            <TypeIcon type={listing.type} className="w-3 h-3 mr-1" />
            {listing.type}
          </span>
        </div>

        <div className="absolute bottom-2.5 right-2.5 bg-white/95 backdrop-blur-sm
                        rounded-lg px-2.5 py-1 shadow-sm">
          <span className="text-xs font-bold text-neutral-800">
            ${listing.price_per_night}
            <span className="font-normal text-neutral-500">/night</span>
          </span>
        </div>
      </div>

      <div className="p-3.5 flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-neutral-900 leading-snug line-clamp-2 group-hover:text-airbnb-500 transition-colors">
          {listing.name}
        </h3>

        <div className="flex items-start gap-1 text-xs text-neutral-500">
          <MapPin className="w-3 h-3 mt-0.5 flex-shrink-0 text-airbnb-400" />
          <span className="line-clamp-1">{listing.location}</span>
        </div>

        <div className="flex items-center gap-1">
          <Star className="w-3.5 h-3.5 fill-airbnb-500 text-airbnb-500" />
          <span className="text-xs font-semibold text-neutral-800">{listing.rating}</span>
          <span className="text-xs text-neutral-400 ml-0.5">Superhost</span>
        </div>

        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {topAmenities.map((a) => (
            <span key={a} className="amenity-pill">
              <AmenityIcon name={a} />
              {a}
            </span>
          ))}
        </div>

        <button className="mt-1 w-full flex items-center justify-center gap-1.5 py-2 rounded-xl
                           text-xs font-semibold text-airbnb-500 border border-airbnb-200
                           hover:bg-airbnb-50 transition-colors duration-200 group/btn">
          View details
          <ChevronRight className="w-3.5 h-3.5 group-hover/btn:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  )
}

function ListingsRow({ listings }) {
  if (!listings?.length) return null
  return (
    <div className="mt-3 -mx-1">
      <div className="flex gap-3 overflow-x-auto pb-2 px-1 snap-x snap-mandatory
                      scrollbar-thin scrollbar-thumb-neutral-200">
        {listings.map((l) => (
          <div key={l.id} className="snap-start">
            <ListingCard listing={l} />
          </div>
        ))}
      </div>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="bubble-user">{msg.text}</div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-3 animate-slide-up">
      <AiAvatar />
      <div className="flex-1 min-w-0">
        <div className={`bubble-ai prose prose-sm max-w-none${msg.isError ? ' border border-red-200 bg-red-50' : ''}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {msg.text || ''}
          </ReactMarkdown>
          {msg.streaming && <StreamCursor />}
        </div>
        {!msg.streaming && msg.listings?.length > 0 && <ListingsRow listings={msg.listings} />}
      </div>
    </div>
  )
}

function WelcomeHero({ onChip }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 hero-gradient animate-fade-in">
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg mb-5"
           style={{ background: 'linear-gradient(135deg,#FF385C,#E61E4D)' }}>
        <Sparkles className="w-8 h-8 text-white" />
      </div>

      <h1 className="text-2xl sm:text-3xl font-bold text-center text-neutral-900 mb-2 tracking-tight">
        Find your perfect stay
      </h1>
      <p className="text-sm sm:text-base text-center text-neutral-500 mb-8 max-w-sm leading-relaxed">
        Describe your dream getaway and our AI will curate the best listings from around the world for you.
      </p>

      <div className="flex flex-wrap gap-2.5 justify-center max-w-lg">
        {QUICK_CHIPS.map((c) => (
          <button key={c.label} id={`chip-${c.label.replace(/\s+/g, '-').toLowerCase()}`}
                  className="chip" onClick={() => onChip(c.label)}>
            <span>{c.emoji}</span>
            <span>{c.label}</span>
          </button>
        ))}
      </div>

      <div className="mt-10 flex items-center gap-6 text-xs text-neutral-400">
        {[['50+', 'hand-picked stays'], ['30+', 'countries'], ['4.9★', 'avg rating']].map(([n, l]) => (
          <div key={l} className="flex flex-col items-center gap-0.5">
            <span className="text-sm font-bold text-neutral-700">{n}</span>
            <span>{l}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  
  // Settings modal states
  const [showSettings, setShowSettings] = useState(false)
  const [apiKeyInput, setApiKeyInput]   = useState('')
  const [showKey, setShowKey]           = useState(false)
  const [quotaExhausted, setQuotaExhausted] = useState(false)

  const messagesEndRef = useRef(null)
  const textareaRef    = useRef(null)
  const inputWrapRef   = useRef(null)
  const abortRef       = useRef(null)

  const hasMessages = messages.length > 0

  // Load custom API key on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('user_gemini_key') || ''
    setApiKeyInput(savedKey)
  }, [])

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }, [input])

  const handleSaveKey = (e) => {
    e.preventDefault()
    const trimmed = apiKeyInput.trim()
    if (trimmed) {
      localStorage.setItem('user_gemini_key', trimmed)
    } else {
      localStorage.removeItem('user_gemini_key')
    }
    setQuotaExhausted(false)
    setShowSettings(false)
  }

  const sendMessage = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setError(null)
    setInput('')

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', text: trimmed }])
    setLoading(true)

    // Add placeholder AI message
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', text: '', listings: [], streaming: true },
    ])

    let accText = ''
    let finalListings = []

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const url = `${STREAM_URL}?message=${encodeURIComponent(trimmed)}`
      const headers = { 'Content-Type': 'application/json' }
      
      // Attach user key if it exists in local storage
      const userKey = localStorage.getItem('user_gemini_key')
      if (userKey && userKey.trim()) {
        headers['X-User-API-Key'] = userKey.trim()
      }

      const response = await fetch(url, { 
        signal: controller.signal,
        headers: headers
      })

      if (!response.ok) {
        let detail = `Server error ${response.status}`
        try {
          const body = await response.json()
          if (body?.detail) detail = body.detail
        } catch { /* ignore */ }
        
        // If server tells us 429 (quota) or 400 (no key), trigger quota flow
        if (response.status === 429 || (response.status === 400 && detail.includes("key"))) {
          setQuotaExhausted(true)
          setShowSettings(true)
        }
        throw new Error(detail)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data:')) continue

          const jsonStr = line.slice(5).trim()
          let event
          try { event = JSON.parse(jsonStr) } catch { continue }

          if (event.type === 'token') {
            accText += event.value
            setMessages((prev) => {
              const next = [...prev]
              next[next.length - 1] = {
                ...next[next.length - 1],
                text: accText,
                streaming: true,
              }
              return next
            })
          } else if (event.type === 'listings') {
            finalListings = event.value
          } else if (event.type === 'done') {
            setMessages((prev) => {
              const next = [...prev]
              next[next.length - 1] = {
                ...next[next.length - 1],
                text: accText,
                listings: finalListings,
                streaming: false,
              }
              return next
            })
          } else if (event.type === 'error') {
            // Check if error contains quota signals
            const isQuota = event.value.includes("quota") || event.value.includes("429") || event.value.includes("exhausted") || event.value.includes("RESOURCE_EXHAUSTED")
            if (isQuota) {
              setQuotaExhausted(true)
              setShowSettings(true)
            }
            throw new Error(event.value)
          }
        }
      }

      // Safety finalisation
      setMessages((prev) => {
        const next = [...prev]
        if (next[next.length - 1]?.streaming) {
          next[next.length - 1] = {
            ...next[next.length - 1],
            listings: finalListings,
            streaming: false,
          }
        }
        return next
      })

    } catch (e) {
      if (e.name === 'AbortError') {
        setMessages((prev) => {
          const next = [...prev]
          if (next[next.length - 1]?.streaming) {
            next[next.length - 1] = { ...next[next.length - 1], streaming: false }
          }
          return next
        })
        return
      }

      const msg = e.message ?? 'Unknown error'
      console.error("Concierge API Error:", e)

      const isOverloaded = msg.includes('unavailable') || msg.includes('503')
      const isQuotaError = msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('429') || msg.includes('key')
      const isNetworkErr = msg.includes('Cannot reach') || msg.includes('Failed to fetch') || msg.includes('NetworkError')

      const aiText = isQuotaError
        ? '⚠️ **Shared AI limit reached.**\n\nOur shared search quota is temporarily exhausted. To continue chatting, please click the **Settings ⚙️** icon in the header and add your own free Gemini API key.\n\n*Your key is stored safely in your browser and is never saved on our servers.*'
        : isOverloaded
          ? 'The AI concierge is currently experiencing high demand. Please try sending your message again in a few seconds. ⏳'
          : isNetworkErr
            ? "I'm having trouble connecting to the concierge service right now. Please check your internet connection or try again in a moment. 🌐"
            : 'Oops! I encountered an unexpected issue while processing your request. Please try again in a moment.'

      setMessages((prev) => {
        const next = [...prev]
        if (next.length > 0 && next[next.length - 1].role === 'assistant' && next[next.length - 1].streaming) {
          next[next.length - 1] = { role: 'assistant', text: aiText, listings: [], streaming: false, isError: true }
        } else {
          next.push({ role: 'assistant', text: aiText, listings: [], streaming: false, isError: true })
        }
        return next
      })
    } finally {
      setLoading(false)
      abortRef.current = null
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }, [loading, messages.length])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const clearChat = () => {
    abortRef.current?.abort()
    setMessages([])
    setError(null)
    textareaRef.current?.focus()
  }

  return (
    <div className="flex flex-col h-dvh bg-neutral-50">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 glass-card border-b border-neutral-100 shadow-sm">
        <div className="max-w-3xl mx-auto flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                 style={{ background: 'linear-gradient(135deg,#FF385C,#E61E4D)' }}>
              <svg viewBox="0 0 24 24" fill="white" className="w-5 h-5">
                <path d="M12.005 2C9.205 2 7 4.206 7 7.005c0 3.312 3.267 7.73 4.545 9.395.406.524 1.107.524 1.514 0C14.337 14.736 17 10.316 17 7.005 17 4.206 14.805 2 12.005 2zm0 6.5A1.5 1.5 0 1 1 12 5a1.5 1.5 0 0 1 .005 3.5z"/>
              </svg>
            </div>
            <div>
              <span className="text-base font-bold logo-text tracking-tight">airbnb</span>
              <span className="ml-1.5 text-xs text-neutral-400 font-medium hidden sm:inline">AI Concierge</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full
                            bg-emerald-50 border border-emerald-100">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-medium text-emerald-700">AI Online</span>
            </div>

            {/* Settings button */}
            <button
              onClick={() => {
                setQuotaExhausted(false)
                setShowSettings(true)
              }}
              title="API Key Settings"
              className="flex items-center justify-center p-2 rounded-xl text-neutral-500 border border-neutral-200 bg-white hover:text-neutral-800 hover:border-neutral-300 hover:shadow-sm transition-all duration-200"
            >
              <Settings className="w-4 h-4" />
            </button>

            {hasMessages && (
              <button id="clear-chat-btn"
                      onClick={clearChat}
                      title="Clear chat"
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium
                                 text-neutral-500 border border-neutral-200 bg-white
                                 hover:text-neutral-800 hover:border-neutral-300 transition-all duration-200">
                <X className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Chat area ─────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto chat-scroll">
        <div className="max-w-3xl mx-auto px-4 py-6">

          {!hasMessages ? (
            <WelcomeHero onChip={(label) => sendMessage(label)} />
          ) : (
            <div className="flex flex-col gap-5">
              {messages.map((msg, i) => (
                <Message key={i} msg={msg} />
              ))}

              {loading && messages[messages.length - 1]?.text === '' && (
                <TypingIndicator />
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* ── Input area ────────────────────────────────────────────────────── */}
      <div className="sticky bottom-0 z-20 glass-card border-t border-neutral-100 shadow-[0_-4px_20px_rgba(0,0,0,0.06)]">
        <div className="max-w-3xl mx-auto px-4 py-3">

          {hasMessages && (
            <div className="flex gap-2 overflow-x-auto pb-2 mb-2 scrollbar-none">
              {QUICK_CHIPS.map((c) => (
                <button key={c.label}
                        className="chip text-xs py-1.5 px-3 flex-shrink-0"
                        onClick={() => sendMessage(c.label)}>
                  <span>{c.emoji}</span>
                  <span className="hidden sm:inline">{c.label}</span>
                </button>
              ))}
            </div>
          )}

          <div ref={inputWrapRef}
               className="flex items-end gap-3 bg-white rounded-2xl border border-neutral-200
                          shadow-sm px-4 py-3 focus-within:border-airbnb-400
                          focus-within:shadow-[0_0_0_3px_rgba(255,56,92,0.12)]
                          transition-all duration-200">
            <textarea
              id="chat-input"
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything — beach villa, mountain cabin, city studio…"
              rows={1}
              className="chat-input"
              disabled={loading}
              aria-label="Type your travel query"
            />

            <button
              id="send-btn"
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || loading}
              className="send-btn flex-shrink-0"
              aria-label="Send message"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>

          <p className="text-center text-xs text-neutral-400 mt-2 leading-relaxed">
            Powered by&nbsp;<span className="font-medium text-neutral-500">Gemini AI</span>
            &nbsp;·&nbsp;Press <kbd className="px-1 py-0.5 rounded bg-neutral-100 font-mono text-[10px]">Enter</kbd> to send
            &nbsp;·&nbsp;<kbd className="px-1 py-0.5 rounded bg-neutral-100 font-mono text-[10px]">Shift+Enter</kbd> for new line
          </p>
          <p className="text-center text-[10px] text-neutral-400 mt-2.5 leading-relaxed max-w-md mx-auto border-t border-neutral-100/60 pt-2">
            Disclaimer: This is an independent portfolio project created for demonstration purposes. It is not affiliated with or endorsed by Airbnb.
          </p>
        </div>
      </div>

      {/* ── Settings Modal ─────────────────────────────────────────────────── */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md bg-white rounded-3xl p-6 border border-neutral-100 shadow-2xl relative">
            <button
              onClick={() => setShowSettings(false)}
              className="absolute top-4 right-4 p-1.5 rounded-full hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-2.5 mb-4">
              <div className="p-2 rounded-xl bg-airbnb-50 text-airbnb-500">
                <Key className="w-5 h-5" />
              </div>
              <h2 className="text-lg font-bold text-neutral-900">
                {quotaExhausted ? "Limit Reached" : "API Settings"}
              </h2>
            </div>

            {quotaExhausted && (
              <div className="mb-4 p-3 bg-amber-50 border border-amber-100 rounded-2xl text-xs text-amber-800 leading-relaxed">
                ⚠️ Our shared Gemini API quota has been exhausted. You can supply your own API key to continue chatting immediately.
              </div>
            )}

            <form onSubmit={handleSaveKey} className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-2">
                  Your Gemini API Key (Optional)
                </label>
                <div className="relative flex items-center">
                  <input
                    type={showKey ? "text" : "password"}
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="AIzaSy..."
                    className="w-full bg-neutral-50 border border-neutral-200 rounded-xl py-2.5 pl-3 pr-10 text-sm focus:outline-none focus:border-airbnb-400 focus:bg-white transition-all font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey(!showKey)}
                    className="absolute right-3 text-neutral-400 hover:text-neutral-600 transition-colors"
                  >
                    {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-[11px] text-neutral-400 mt-2 leading-relaxed">
                  Get a free personal API key from the{" "}
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noreferrer"
                    className="text-airbnb-500 hover:underline inline-flex items-center gap-0.5"
                  >
                    Google AI Studio
                  </a>.
                </p>
              </div>

              {/* Safety Assurance Info */}
              <div className="p-3.5 bg-emerald-50/70 border border-emerald-100 rounded-2xl flex gap-3 items-start">
                <ShieldCheck className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                <div className="flex flex-col gap-0.5">
                  <h4 className="text-xs font-semibold text-emerald-800">Your key is 100% safe</h4>
                  <p className="text-[11px] text-emerald-700/90 leading-relaxed">
                    It is stored locally on your device (`localStorage`) and sent securely directly to the Gemini API endpoint. It is never stored on our servers.
                  </p>
                </div>
              </div>

              <div className="flex gap-2.5 mt-2">
                <button
                  type="button"
                  onClick={() => {
                    setApiKeyInput("")
                    localStorage.removeItem('user_gemini_key')
                    setQuotaExhausted(false)
                    setShowSettings(false)
                  }}
                  className="flex-1 py-2.5 rounded-xl border border-neutral-200 text-sm font-semibold text-neutral-500 hover:bg-neutral-50 hover:text-neutral-700 transition-all"
                >
                  Clear Key
                </button>
                <button
                  type="submit"
                  className="flex-1 py-2.5 rounded-xl text-sm font-semibold text-white bg-airbnb-500 hover:bg-airbnb-600 shadow-md hover:shadow-lg transition-all"
                >
                  Save & Continue
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
