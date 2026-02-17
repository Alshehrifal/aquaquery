import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Waves } from 'lucide-react';
import { useChat } from '../hooks/useChat';
import { useAgentActivity } from '../hooks/useAgentActivity';
import { ActivityPanel } from './ActivityPanel';
import { ChatInput } from './ChatInput';
import { MessageBubble } from './MessageBubble';
import { SampleQueries } from './SampleQueries';

function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function ChatInterface() {
  const [sessionId] = useState(generateSessionId);
  const { events, addEvent, addWarning, clearEvents } = useAgentActivity();

  const chatCallbacks = useMemo(
    () => ({ onAgentEvent: addEvent, onWarning: addWarning }),
    [addEvent, addWarning],
  );

  const { messages, isLoading, error, statusText, send, clearError } = useChat(
    sessionId,
    true,
    chatCallbacks,
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(
    (message: string) => {
      clearEvents();
      send(message);
    },
    [send, clearEvents],
  );

  const handleSampleQuery = useCallback(
    (query: string) => {
      handleSend(query);
    },
    [handleSend],
  );

  return (
    <div id="chat" className="flex min-h-screen bg-background">
      {/* Chat section */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border bg-card/80 px-6 py-3 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg"
              style={{ backgroundImage: 'var(--gradient-ocean)' }}
            >
              <Waves className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-display text-lg font-semibold text-foreground">AquaQuery</h1>
              <p className="text-xs text-muted-foreground">Argo Ocean Data Explorer</p>
            </div>
          </div>
        </header>

        {/* Messages area */}
        <div className="custom-scrollbar flex-1 overflow-y-auto bg-background px-6 py-4">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-secondary">
                <Waves className="h-8 w-8 text-ocean-cyan" />
              </div>
              <h2 className="mb-2 font-display text-xl font-semibold text-foreground">
                Welcome to AquaQuery
              </h2>
              <p className="mb-6 max-w-md text-center text-sm text-muted-foreground">
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
          <div className="mx-6 mb-2 flex items-center justify-between rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
            <span>{error}</span>
            <button onClick={clearError} className="ml-2 text-destructive/60 hover:text-destructive">
              Dismiss
            </button>
          </div>
        )}

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          disabled={isLoading}
          statusText={isLoading ? statusText : undefined}
        />
      </div>

      {/* Activity Panel -- hidden on mobile, visible on lg+ */}
      <div className="hidden w-80 lg:flex lg:w-96">
        <ActivityPanel events={events} isLoading={isLoading} />
      </div>
    </div>
  );
}
