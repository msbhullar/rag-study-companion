"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].content = data.answer;
        return updated;
      });
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].content =
          "Error connecting to backend. Make sure the API is running.";
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `✓ ${data.message}` },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Upload failed. Check the backend." },
      ]);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  return (
    <main className="flex flex-col h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-medium">RAG Study Companion</h1>
          <p className="text-sm text-gray-400">Ask questions about your study materials</p>
        </div>
        <label className="cursor-pointer bg-gray-800 hover:bg-gray-700 text-sm px-4 py-2 rounded-lg transition">
          {uploading ? "Uploading..." : "Upload PDF"}
          <input
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={uploadFile}
            disabled={uploading}
          />
        </label>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <p className="text-xl mb-2">Ask anything about your documents</p>
            <p className="text-sm">Upload a PDF or ask about already ingested materials</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-2xl px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              {msg.content || (
                <span className="animate-pulse text-gray-400">Thinking...</span>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 px-4 py-4">
        <div className="flex gap-3 max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask a question about your study materials..."
            className="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm outline-none focus:ring-1 focus:ring-blue-500 placeholder-gray-500"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 px-5 py-3 rounded-xl text-sm font-medium transition"
          >
            {loading ? "Thinking..." : "Send"}
          </button>
        </div>
      </div>
    </main>
  );
}