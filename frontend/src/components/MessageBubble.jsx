export default function MessageBubble({ role, text }) {
    return (
      <div className={`bubble ${role}`}>
        <span className="bubble-label">{role === "user" ? "You" : "ATPInsight"}</span>
        <p>{text}</p>
      </div>
    )
  }