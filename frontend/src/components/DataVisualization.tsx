import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Visualization } from '../types';

interface DataVisualizationProps {
  visualization: Visualization;
}

export function DataVisualization({ visualization }: DataVisualizationProps) {
  const { plotly_json, description, chart_type } = visualization;

  const layout = useMemo(() => {
    if (!plotly_json) return {};
    return {
      ...plotly_json.layout,
      autosize: true,
      margin: { l: 60, r: 30, t: 40, b: 50 },
    };
  }, [plotly_json]);

  if (!plotly_json || !plotly_json.data) {
    return null;
  }

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
      <Plot
        data={plotly_json.data}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `aquaquery_${chart_type}`,
          },
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
      {description && (
        <p className="mt-2 text-center text-sm text-slate-500">{description}</p>
      )}
    </div>
  );
}
