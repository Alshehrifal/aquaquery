import type { ChatMessage, ChatResponse, DatasetMetadata, VariableInfo, Visualization } from '../types';

const API_BASE = '/api/v1';

export async function sendMessage(
  sessionId: string,
  message: string,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export function streamMessage(
  sessionId: string,
  message: string,
  callbacks: {
    onStatus?: (agent: string, action: string) => void;
    onToken?: (content: string) => void;
    onVisualization?: (viz: Visualization) => void;
    onWarning?: (message: string) => void;
    onDone?: (messageId: string, agentPath: string[]) => void;
    onError?: (message: string) => void;
  },
): () => void {
  const params = new URLSearchParams({
    session_id: sessionId,
    message,
  });

  const eventSource = new EventSource(`${API_BASE}/chat/stream?${params}`);

  eventSource.addEventListener('status', (e) => {
    const data = JSON.parse(e.data);
    callbacks.onStatus?.(data.agent, data.action);
  });

  eventSource.addEventListener('token', (e) => {
    const data = JSON.parse(e.data);
    callbacks.onToken?.(data.content);
  });

  eventSource.addEventListener('visualization', (e) => {
    const data = JSON.parse(e.data);
    callbacks.onVisualization?.(data);
  });

  eventSource.addEventListener('warning', (e) => {
    const data = JSON.parse(e.data);
    callbacks.onWarning?.(data.message);
  });

  eventSource.addEventListener('done', (e) => {
    const data = JSON.parse(e.data);
    callbacks.onDone?.(data.message_id, data.agent_path);
    eventSource.close();
  });

  eventSource.addEventListener('error', (e) => {
    if (e instanceof MessageEvent) {
      const data = JSON.parse(e.data);
      callbacks.onError?.(data.message);
    } else {
      callbacks.onError?.('Connection lost');
    }
    eventSource.close();
  });

  return () => eventSource.close();
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
  if (!response.ok) {
    if (response.status === 404) return [];
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

export async function getVariables(): Promise<VariableInfo[]> {
  const response = await fetch(`${API_BASE}/data/variables`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return data.variables;
}

export async function getMetadata(): Promise<DatasetMetadata> {
  const response = await fetch(`${API_BASE}/data/metadata`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
