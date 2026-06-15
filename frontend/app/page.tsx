// "use client" tells Next.js this component runs in the browser (not on the
// server). It is required for any component that uses interactivity — state,
// event handlers, effects. Our chat page needs all of those.
"use client";

import { useRef, useState } from "react";
import styles from "./page.module.css";

// The backend base URL. NEXT_PUBLIC_ vars are exposed to the browser; we read
// it from .env.local so the same code works locally and in production.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// A TypeScript "type" describing the shape of one chat message. This is the
// core benefit of TS: the compiler checks we never build a malformed message.
type Message = {
  role: "user" | "assistant";
  content: string;
};

const EXAMPLE_PROMPTS = [
  "How do I forward the desk phone?",
  "How do I log a package into the mailroom system?",
  "Can I give a resident's room number to a parent?",
];

export default function Home() {
  // useState gives us reactive variables: when we call the setter, React
  // re-renders the UI. messages = the conversation; input = the text box;
  // isStreaming = whether a response is currently arriving.
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  // A ref to the scroll container so we can auto-scroll to the newest message.
  const listRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
    });
  }

  async function sendMessage(text: string) {
    const question = text.trim();
    if (!question || isStreaming) return;

    // Optimistically add the user message + an empty assistant message that we
    // will fill in token-by-token as the stream arrives.
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setIsStreaming(true);
    scrollToBottom();

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Backend returned ${res.status}`);
      }

      // The backend streams Server-Sent Events. We read the response body as a
      // stream of bytes, decode to text, and parse out the "data:" lines,
      // appending each token to the last (assistant) message.
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by newlines; each data line starts with
        // "data:". We process complete lines and keep any partial tail.
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data:")) {
            const token = line.slice(5).replace(/^ /, "");
            appendToLastAssistant(token);
            scrollToBottom();
          }
        }
      }
    } catch (err) {
      appendToLastAssistant(
        `\n\n⚠️ Could not reach the assistant. Is the backend running? (${
          err instanceof Error ? err.message : "unknown error"
        })`,
      );
    } finally {
      setIsStreaming(false);
    }
  }

  // Append text to the most recent assistant message immutably (React state
  // must never be mutated in place).
  function appendToLastAssistant(token: string) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = { ...last, content: last.content + token + " " };
      }
      return next;
    });
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>🛎️ Center Desk Assistant</h1>
        <p>Ask about residence hall Center Desk procedures.</p>
      </header>

      <div className={styles.chat} ref={listRef}>
        {messages.length === 0 ? (
          <div className={styles.empty}>
            <p>Try one of these:</p>
            <div className={styles.examples}>
              {EXAMPLE_PROMPTS.map((p) => (
                <button
                  key={p}
                  className={styles.example}
                  onClick={() => sendMessage(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`${styles.bubble} ${
                m.role === "user" ? styles.user : styles.assistant
              }`}
            >
              {m.content || (isStreaming ? "…" : "")}
            </div>
          ))
        )}
      </div>

      <form
        className={styles.inputRow}
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage(input);
        }}
      >
        <input
          className={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask here…"
          disabled={isStreaming}
        />
        <button
          className={styles.send}
          type="submit"
          disabled={isStreaming || !input.trim()}
        >
          {isStreaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
