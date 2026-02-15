import { useCallback, useRef, useState } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  statusText?: string;
}

export function ChatInput({ onSend, disabled, statusText }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [input, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
  }, []);

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      {statusText && (
        <div className="mb-2 flex items-center gap-2 text-sm text-[var(--ocean-mid)]">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--ocean-light)]" />
          {statusText}
        </div>
      )}
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Ask about ocean data..."
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:border-[var(--ocean-light)] focus:outline-none focus:ring-1 focus:ring-[var(--ocean-light)] disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !input.trim()}
          className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--ocean-mid)] text-white transition-colors hover:bg-[var(--ocean-dark)] disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Send message"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
            <path d="M3.105 2.288a.75.75 0 0 0-.826.95l1.414 4.926A1.5 1.5 0 0 0 5.135 9.25h6.115a.75.75 0 0 1 0 1.5H5.135a1.5 1.5 0 0 0-1.442 1.086l-1.414 4.926a.75.75 0 0 0 .826.95 28.897 28.897 0 0 0 15.293-7.155.75.75 0 0 0 0-1.114A28.897 28.897 0 0 0 3.105 2.288Z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
