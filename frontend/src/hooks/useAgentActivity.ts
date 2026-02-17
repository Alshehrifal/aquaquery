import { useCallback, useState } from 'react';

export interface AgentEvent {
  readonly id: string;
  readonly agent: string;
  readonly action: string;
  readonly timestamp: number;
  readonly status: 'active' | 'completed';
}

interface UseAgentActivityReturn {
  readonly events: readonly AgentEvent[];
  readonly addEvent: (agent: string, action: string) => void;
  readonly clearEvents: () => void;
}

let eventCounter = 0;

function generateEventId(): string {
  eventCounter += 1;
  return `evt-${eventCounter}-${Date.now()}`;
}

export function useAgentActivity(): UseAgentActivityReturn {
  const [events, setEvents] = useState<AgentEvent[]>([]);

  const addEvent = useCallback((agent: string, action: string) => {
    if (agent === 'done') {
      // Mark all active events as completed
      setEvents((prev) =>
        prev.map((evt) =>
          evt.status === 'active' ? { ...evt, status: 'completed' as const } : evt,
        ),
      );
      return;
    }

    const newEvent: AgentEvent = {
      id: generateEventId(),
      agent,
      action,
      timestamp: Date.now(),
      status: 'active',
    };

    setEvents((prev) => {
      // Mark previous active event as completed, then append new one
      const updated = prev.map((evt) =>
        evt.status === 'active' ? { ...evt, status: 'completed' as const } : evt,
      );
      return [...updated, newEvent];
    });
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  return { events, addEvent, clearEvents };
}
