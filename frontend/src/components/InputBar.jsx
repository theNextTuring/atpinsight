import { useState } from "react"

export default function InputBar({ onSubmit, loading }) {
  const [value, setValue] = useState("")

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSubmit(value)
      setValue("")
    }
  }

  function handleClick() {
    onSubmit(value)
    setValue("")
  }

  return (
    <div className="input-bar">
      <input
        type="text"
        placeholder="Ask about ATP players, matches, tournaments..."
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <button onClick={handleClick} disabled={loading || !value.trim()}>
        {loading ? "..." : "Ask"}
      </button>
    </div>
  )
}