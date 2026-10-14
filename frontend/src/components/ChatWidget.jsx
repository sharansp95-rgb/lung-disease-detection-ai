import { useEffect, useRef, useState } from "react";
import { sendChatMessage, ApiError } from "../api";
import { useAuth } from "../AuthContext";

// Mirror the backend's limits (backend/schemas.py, backend/chat.py): the
// server only ever uses the last 10 turns, so sending more is wasted payload.
const MAX_HISTORY_TURNS = 10;
const MAX_MESSAGE_CHARS = 2000;

const FAQ_QUESTIONS = [
  "What does the heatmap show?",
  "How accurate is this model?",
  "Is this a medical diagnosis?",
];

export default function ChatWidget({ predictionContext }) {
  const { token } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]); // {role: 'user'|'assistant', content, source?}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading, isOpen]);

  async function handleSend(textOverride) {
    const trimmed = (typeof textOverride === "string" ? textOverride : input).trim();
    if (!trimmed || loading) return;

    const userMessage = { role: "user", content: trimmed };
    const updated = [...messages, userMessage];
    setMessages(updated);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const historyForApi = messages
        .slice(-MAX_HISTORY_TURNS)
        .map((m) => ({ role: m.role, content: m.content }));
      const res = await sendChatMessage(trimmed, historyForApi, predictionContext, token);
      setMessages([...updated, { role: "assistant", content: res.reply, source: res.source }]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) {
        setError("Please verify your email to use the chat -- check the link we sent you.");
      } else {
        setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <>
      <button
        className="chat-fab"
        onClick={() => setIsOpen((v) => !v)}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? (
          <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2.2">
            <line x1="6" y1="6" x2="18" y2="18" strokeLinecap="round" />
            <line x1="18" y1="6" x2="6" y2="18" strokeLinecap="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
            <path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-5 4V6a2 2 0 0 1 2-2z" />
          </svg>
        )}
      </button>

      {isOpen && (
        <div className="chat-panel">
          <div className="chat-panel-header">
            <span>{predictionContext ? "Ask about this result" : "Ask about this tool"}</span>
            <button className="chat-close-btn" onClick={() => setIsOpen(false)} aria-label="Close chat">
              ×
            </button>
          </div>

          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`chat-message ${m.role}`}>
                <div className="chat-bubble">{m.content}</div>
                {m.role === "assistant" && m.source && (
                  <span className="chat-source">{m.source === "groq" ? "Groq" : "fallback"}</span>
                )}
              </div>
            ))}

            {loading && (
              <div className="chat-message assistant">
                <div className="chat-bubble typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}

            {error && <div className="chat-error">{error}</div>}
            <div ref={bottomRef} />
          </div>

          {messages.length === 0 && (
            <div className="chat-suggestions">
              {[...FAQ_QUESTIONS, ...(predictionContext ? ["Why did it predict this result?"] : [])].map(
                (q) => (
                  <button
                    key={q}
                    type="button"
                    className="chat-chip"
                    onClick={() => handleSend(q)}
                    disabled={loading}
                  >
                    {q}
                  </button>
                )
              )}
            </div>
          )}

          <div className="chat-input-row">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question"
              maxLength={MAX_MESSAGE_CHARS}
              disabled={loading}
              rows={1}
            />
            <button onClick={() => handleSend()} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
