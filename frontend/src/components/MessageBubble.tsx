import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sanitizeContent } from '../services/sanitize';
import type { ChatMessage } from '../types';
import { DataVisualization } from './DataVisualization';

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const cleanContent = useMemo(
    () => (isUser ? message.content : sanitizeContent(message.content)),
    [isUser, message.content],
  );

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-[var(--ocean-mid)] text-white'
            : 'bg-slate-100 text-slate-800'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{cleanContent}</p>
        ) : (
          <>
            <div className="prose prose-sm max-w-none prose-headings:text-slate-800 prose-p:text-slate-700 prose-a:text-[var(--ocean-mid)]">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {cleanContent}
              </ReactMarkdown>
            </div>

            {message.visualization?.plotly_json && (
              <DataVisualization visualization={message.visualization} />
            )}

            {message.sources && message.sources.length > 0 && (
              <div className="mt-2 border-t border-slate-200 pt-2">
                <p className="text-xs text-slate-400">
                  Sources: {message.sources.join(', ')}
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
