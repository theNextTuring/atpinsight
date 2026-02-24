import { useState } from "react"
import ChatWindow from "./components/ChatWindow"
import InputBar from "./components/InputBar"
import "./index.css"

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Welcome to ATPInsight. Ask me anything about ATP matches, players, and performance in 2024."
    }
  ])
  const [loading, setLoading] = useState(false)

  async function handleSubmit(question) {
    if (!question.trim()) return

    setMessages(prev => [...prev, { role: "user", text: question }])
    setLoading(true)

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: "assistant", text: data.answer }])
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", text: "Error connecting to backend. Make sure the server is running." }])
    }

    setLoading(false)
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