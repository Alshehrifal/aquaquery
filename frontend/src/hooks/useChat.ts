import { useCallback, useRef, useState } from 'react';
import { sendMessage, streamMessage } from '../services/api';
import type { ChatMessage, Visualization } from '../types';

interface UseChatReturn {
  messages: readonly ChatMessage[];
  isLoading: boolean;
  error: string | null;
  statusText: string;
  send: (message: string) => void;
  clearError: () => void;
}

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function useChat(sessionId: string, useStreaming = true): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState('');
  const closeRef = useRef<(() => void) | null>(null);

  const addMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const send = useCallback(
    (message: string) => {
      if (!message.trim() || isLoading) return;

      setError(null);
      setIsLoading(true);

      const userMsg: ChatMessage = {
        id: generateId(),
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMsg);

      if (useStreaming) {
        const assistantId = generateId();
        let accumulatedContent = '';
        let visualization: Visualization | null = null;

        // Add placeholder assistant message
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: '', timestamp: new Date().toISOString() },
        ]);

        closeRef.current = streamMessage(sessionId, message, {
          onStatus: (_agent, action) => {
            setStatusText(action);
          },
          onToken: (content) => {
            accumulatedContent += content;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: accumulatedContent } : m,
              ),
            );
          },
          onVisualization: (viz) => {
            visualization = viz;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, visualization: viz } : m,
              ),
            );
          },
          onDone: (_messageId, _agentPath) => {
            setIsLoading(false);
            setStatusText('');
          },
          onError: (errMsg) => {
            setError(errMsg);
            setIsLoading(false);
            setStatusText('');
          },
        });
      } else {
        sendMessage(sessionId, message)
          .then((response) => {
            const assistantMsg: ChatMessage = {
              id: response.message_id,
              role: 'assistant',
              content: response.content,
              visualization: response.visualization,
              sources: response.sources,
              timestamp: response.timestamp,
            };
            addMessage(assistantMsg);
          })
          .catch((err) => {
            setError(err.message);
          })
          .finally(() => {
            setIsLoading(false);
            setStatusText('');
          });
      }
    },
    [sessionId, isLoading, useStreaming, addMessage],
  );

  const clearError = useCallback(() => setError(null), []);

  return { messages, isLoading, error, statusText, send, clearError };
}
