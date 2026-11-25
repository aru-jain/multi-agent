import { useState } from "react";
import { ChevronDown, ChevronRight, Terminal } from "lucide-react";
import type { TraceEntry } from "../types/workflow";

interface TraceTimelineProps {
  entries: TraceEntry[];
}

export function TraceTimeline({ entries }: TraceTimelineProps) {
  const [filter, setFilter] = useState<string>("all");
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const types = Array.from(new Set(entries.map((e) => e.entry_type)));
  const visible =
    filter === "all"
      ? entries
      : entries.filter((e) => e.entry_type === filter);

  const toggleExpand = (idx: number) => {
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <Terminal size={17} className="icon-accent" />
          <h2>Execution Trace</h2>
          <span className="count-pill">{entries.length}</span>
        </div>

        <select
          className="filter-select"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">All events</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {visible.length === 0 ? (
        <div className="empty-state">No trace events recorded yet.</div>
      ) : (
        <div className="trace-list">
          {visible.map((entry, idx) => {
            const hasMeta =
              entry.metadata && Object.keys(entry.metadata).length > 0;
            const isOpen = !!expanded[idx];

            return (
              <div key={idx} className={`trace-item trace-${entry.entry_type}`}>
                <div
                  className="trace-row"
                  onClick={() => hasMeta && toggleExpand(idx)}
                  style={{ cursor: hasMeta ? "pointer" : "default" }}
                >
                  <div className="trace-left">
                    {hasMeta ? (
                      isOpen ? (
                        <ChevronDown size={14} />
                      ) : (
                        <ChevronRight size={14} />
                      )
                    ) : (
                      <span className="trace-dot" />
                    )}
                    <span className={`trace-type type-${entry.entry_type}`}>
                      {entry.entry_type}
                    </span>
                    {entry.agent && (
                      <span className="trace-agent">{entry.agent}</span>
                    )}
                    <span className="trace-content">{entry.content}</span>
                  </div>
                  {entry.timestamp && (
                    <span className="trace-time">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>

                {hasMeta && isOpen && (
                  <pre className="trace-meta">
                    {JSON.stringify(entry.metadata, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
