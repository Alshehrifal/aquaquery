import { motion, AnimatePresence } from 'framer-motion';
import { Activity, CheckCircle, Loader } from 'lucide-react';
import type { AgentEvent } from '../hooks/useAgentActivity';

const AGENT_LABELS: Record<string, string> = {
  supervisor: 'Supervisor',
  rag_agent: 'RAG Agent',
  rag: 'RAG Agent',
  query_agent: 'Query Agent',
  query: 'Query Agent',
  query_for_viz: 'Query Agent',
  viz_agent: 'Viz Agent',
  viz: 'Viz Agent',
  clarify: 'Clarify',
};

interface ActivityPanelProps {
  readonly events: readonly AgentEvent[];
  readonly isLoading: boolean;
}

function EventItem({ event }: { readonly event: AgentEvent }) {
  const isActive = event.status === 'active';
  const label = AGENT_LABELS[event.agent] ?? event.agent;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25 }}
      className="flex items-start gap-3 py-2"
    >
      <div className="mt-0.5 flex-shrink-0">
        {isActive ? (
          <Loader className="h-4 w-4 animate-spin text-ocean-cyan" />
        ) : (
          <CheckCircle className="h-4 w-4 text-emerald-400" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <p className="truncate text-xs text-muted-foreground">{event.action}</p>
      </div>
    </motion.div>
  );
}

export function ActivityPanel({ events, isLoading }: ActivityPanelProps) {
  return (
    <div className="flex h-full flex-col border-l border-border bg-card/50">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Activity className="h-4 w-4 text-ocean-cyan" />
        <h2 className="font-display text-sm font-semibold text-foreground">
          Agent Activity
        </h2>
        {isLoading && (
          <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-ocean-cyan" />
        )}
      </div>

      {/* Timeline */}
      <div className="custom-scrollbar flex-1 overflow-y-auto px-4 py-3">
        {events.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <Activity className="mb-2 h-8 w-8 text-muted-foreground/30" />
            <p className="text-xs text-muted-foreground">
              Agent activity will appear here when you send a query.
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            <AnimatePresence mode="popLayout">
              {events.map((event) => (
                <EventItem key={event.id} event={event} />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
