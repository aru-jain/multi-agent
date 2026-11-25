import { Coins, Cpu, Footprints, Wrench } from "lucide-react";
import type { BudgetUsage } from "../types/workflow";

interface BudgetPanelProps {
  stepCount: number;
  maxSteps: number;
  budget: BudgetUsage;
  maxTokens: number;
  maxCostUsd: number;
}

export function BudgetPanel({
  stepCount,
  maxSteps,
  budget,
  maxTokens,
  maxCostUsd,
}: BudgetPanelProps) {
  const tokens = budget.total_tokens ?? 0;
  const cost = budget.total_cost_usd ?? 0;
  const toolCalls = budget.tool_calls ?? 0;

  const stepPct = Math.min(100, Math.round((stepCount / Math.max(1, maxSteps)) * 100));
  const tokenPct = Math.min(100, Math.round((tokens / Math.max(1, maxTokens)) * 100));
  const costPct = Math.min(100, Math.round((cost / Math.max(0.01, maxCostUsd)) * 100));

  return (
    <div className="card">
      <div className="card-header">
        <h2>Budget &amp; Safety Guards</h2>
      </div>

      <div className="metrics-grid">
        <div className="metric-box">
          <div className="metric-label">
            <Footprints size={14} />
            Steps
          </div>
          <div className="metric-value">
            {stepCount} <span className="muted">/ {maxSteps}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${stepPct}%` }} />
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">
            <Cpu size={14} />
            Tokens
          </div>
          <div className="metric-value">
            {tokens.toLocaleString()}{" "}
            <span className="muted">/ {maxTokens.toLocaleString()}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${tokenPct}%` }} />
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">
            <Coins size={14} />
            Cost (USD)
          </div>
          <div className="metric-value">
            ${cost.toFixed(4)} <span className="muted">/ ${maxCostUsd.toFixed(2)}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${costPct}%` }} />
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">
            <Wrench size={14} />
            Tool Calls
          </div>
          <div className="metric-value">{toolCalls}</div>
        </div>
      </div>
    </div>
  );
}
