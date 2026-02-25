import { useState } from "react"
import ChatWindow from "./components/ChatWindow"
import InputBar from "./components/InputBar"
import "./index.css"

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

export default function App() {
  const [accessCode, setAccessCode] = useState("")
  const [authenticated, setAuthenticated] = useState(false)
  const [authError, setAuthError] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Welcome to ATPInsight. Ask me anything about ATP matches, players, and performance in 2024."
    }
  ])
  const [loading, setLoading] = useState(false)

  async function handleAuth() {
    try {
      const res = await fetch(`${API_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ access_code: accessCode })
      })
      if (res.status === 401) {
        setAuthError(true)
      } else {
        setAuthenticated(true)
        setAuthError(false)
      }
    } catch {
      setAuthError(true)
    }
  }

  async function handleSubmit(question) {
    if (!question.trim()) return
    setMessages(prev => [...prev, { role: "user", text: question }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, access_code: accessCode })
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: "assistant", text: data.answer }])
    } catch {
      setMessages(prev => [...prev, { role: "assistant", text: "Error connecting to backend." }])
    }
    setLoading(false)
  }

  if (!authenticated) {
    return (
      <div className="auth-screen">
        <div className="auth-box">
          <div className="logo">🎾 ATPInsight</div>
          <p className="auth-subtitle">Enter access code to continue</p>
          <input
            type="password"
            placeholder="Access code"
            value={accessCode}
            onChange={e => setAccessCode(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAuth()}
            className="auth-input"
          />
          {authError && <p className="auth-error">Invalid access code</p>}
          <button onClick={handleAuth} className="auth-button">Enter</button>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">🎾 ATPInsight</div>
        <span className="subtitle">2024 ATP Match Intelligence</span>
      </header>
      <ChatWindow messages={messages} loading={loading} />
      <InputBar onSubmit={handleSubmit} loading={loading} />
    </div>
  )
}