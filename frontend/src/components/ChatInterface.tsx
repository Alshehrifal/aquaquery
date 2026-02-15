import { useCallback, useEffect, useRef, useState } from 'react';
import { useChat } from '../hooks/useChat';
import { ChatInput } from './ChatInput';
import { MessageBubble } from './MessageBubble';
import { SampleQueries } from './SampleQueries';

function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function ChatInterface() {
  const [sessionId] = useState(generateSessionId);
  const { messages, isLoading, error, statusText, send, clearError } = useChat(sessionId, true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSampleQuery = useCallback(
    (query: string) => {
      send(query);
    },
    [send],
  );

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--ocean-mid)]">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5 text-white">
              <path d="M12.378 1.602a.75.75 0 0 0-.756 0L3 6.632l9 5.25 9-5.25-8.622-5.03ZM21.75 7.93l-9 5.25v9l8.628-5.032a.75.75 0 0 0 .372-.648V7.93ZM11.25 22.18v-9l-9-5.25v8.57a.75.75 0 0 0 .372.648l8.628 5.033Z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-800">AquaQuery</h1>
            <p className="text-xs text-slate-400">Argo Ocean Data Explorer</p>
          </div>
        </div>
      </header>

      {/* Messages area */}
      <div className="custom-scrollbar flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-[var(--ocean-foam)]">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="h-8 w-8 text-[var(--ocean-mid)]">
                <path fillRule="evenodd" d="M4.848 2.771A49.144 49.144 0 0 1 12 2.25c2.43 0 4.817.178 7.152.52 1.978.29 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.68-3.348 3.97a48.901 48.901 0 0 1-3.476.383.39.39 0 0 0-.297.17l-2.755 4.133a.75.75 0 0 1-1.248 0l-2.755-4.133a.39.39 0 0 0-.297-.17 48.9 48.9 0 0 1-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97Z" clipRule="evenodd" />
              </svg>
            </div>
            <h2 className="mb-2 text-xl font-semibold text-slate-700">Welcome to AquaQuery</h2>
            <p className="mb-6 max-w-md text-center text-sm text-slate-500">
              Ask questions about Argo oceanographic data. Explore temperature, salinity,
              pressure, and oxygen measurements from the global ocean.
            </p>
            <SampleQueries onSelect={handleSampleQuery} disabled={isLoading} />
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mb-2 flex items-center justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          <span>{error}</span>
          <button onClick={clearError} className="ml-2 text-red-400 hover:text-red-600">
            Dismiss
          </button>
        </div>
      )}

      {/* Input */}
      <ChatInput
        onSend={send}
        disabled={isLoading}
        statusText={isLoading ? statusText : undefined}
      />
    </div>
  );
}
