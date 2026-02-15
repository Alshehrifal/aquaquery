export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  visualization?: Visualization | null;
  sources?: string[];
  timestamp?: string;
}

export interface Visualization {
  chart_type: string;
  plotly_json: PlotlyJSON | null;
  description: string;
}

export interface PlotlyJSON {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
}

export interface ChatResponse {
  session_id: string;
  message_id: string;
  content: string;
  visualization?: Visualization | null;
  sources: string[];
  agent_path: string[];
  timestamp: string;
}

export interface VariableInfo {
  name: string;
  display_name: string;
  unit: string;
  description: string;
  typical_range: [number, number];
}

export interface DatasetMetadata {
  lat_bounds: [number, number];
  lon_bounds: [number, number];
  depth_range: [number, number];
  time_range: [string, string];
  total_profiles: number;
  available_variables: string[];
  data_source: string;
  last_updated: string;
}

export interface SSEEvent {
  event: 'status' | 'token' | 'visualization' | 'done' | 'error';
  data: string;
}
