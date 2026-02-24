import { useEffect, useRef } from "react"
import MessageBubble from "./MessageBubble"

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble key={i} role={msg.role} text={msg.text} />
      ))}
      {loading && (
        <div className="bubble assistant loading">
          <span className="dot" /><span className="dot" /><span className="dot" />
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}