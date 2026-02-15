interface SampleQueriesProps {
  onSelect: (query: string) => void;
  disabled: boolean;
}

const SAMPLE_QUERIES = [
  { label: 'What is the Argo program?', icon: '?' },
  { label: 'Show me temperature at 500m in Atlantic Ocean', icon: 'T' },
  { label: 'Compare salinity: Pacific vs Atlantic', icon: 'S' },
  { label: 'Plot oxygen levels over time at 30N, 140W', icon: 'O' },
  { label: "What's a thermocline?", icon: '~' },
];

export function SampleQueries({ onSelect, disabled }: SampleQueriesProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-slate-500">Try a sample query</h3>
      <div className="flex flex-wrap gap-2">
        {SAMPLE_QUERIES.map((q) => (
          <button
            key={q.label}
            onClick={() => onSelect(q.label)}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-600 transition-colors hover:border-[var(--ocean-light)] hover:bg-[var(--ocean-foam)] hover:text-[var(--ocean-mid)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--ocean-pale)] text-xs font-bold text-[var(--ocean-dark)]">
              {q.icon}
            </span>
            {q.label}
          </button>
        ))}
      </div>
    </div>
  );
}
