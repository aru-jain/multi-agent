import { useState } from "react";
import { Play, SlidersHorizontal, Sparkles } from "lucide-react";
import type { WorkflowRunRequest } from "../types/workflow";

interface WorkflowFormProps {
  onSubmit: (req: WorkflowRunRequest) => Promise<void>;
  isSubmitting: boolean;
}

const PRESET_GOALS = [
  {
    label: "Python Prime Analyzer",
    goal: "Write and execute a Python script to find all prime numbers under 100 and compute their sum and average.",
  },
  {
    label: "Web Research + Code",
    goal: "Research the top 3 Python async web frameworks and write a Python comparison summary script.",
  },
  {
    label: "DB Schema + Write (Approval)",
    goal: "Inspect the database tables, then create an audit_notes table and insert a sample note.",
  },
];

export function WorkflowForm({ onSubmit, isSubmitting }: WorkflowFormProps) {
  const [goal, setGoal] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxSteps, setMaxSteps] = useState(50);
  const [maxTokens, setMaxTokens] = useState(100000);
  const [maxCostUsd, setMaxCostUsd] = useState(5.0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim() || isSubmitting) return;
    await onSubmit({
      goal: goal.trim(),
      max_steps: maxSteps,
      max_tokens: maxTokens,
      max_cost_usd: maxCostUsd,
    });
  };

  return (
    <form className="card form-card" onSubmit={handleSubmit}>
      <div className="card-header">
        <div className="card-title">
          <Sparkles size={18} className="icon-accent" />
          <h2>New Workflow Goal</h2>
        </div>
        <button
          type="button"
          className="btn-ghost btn-sm"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <SlidersHorizontal size={14} />
          {showAdvanced ? "Hide Limits" : "Safety Limits"}
        </button>
      </div>

      <textarea
        className="goal-input"
        rows={3}
        placeholder="Describe a multi-step goal for the Planner, Researcher, Coder, and DB Analyst..."
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        disabled={isSubmitting}
      />

      <div className="presets-row">
        <span className="presets-label">Examples:</span>
        {PRESET_GOALS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            className="chip"
            onClick={() => setGoal(preset.goal)}
            disabled={isSubmitting}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {showAdvanced && (
        <div className="limits-grid">
          <label className="field">
            <span>Max Steps</span>
            <input
              type="number"
              min={1}
              max={200}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value) || 50)}
            />
          </label>
          <label className="field">
            <span>Max Tokens</span>
            <input
              type="number"
              min={1000}
              step={1000}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value) || 100000)}
            />
          </label>
          <label className="field">
            <span>Max Cost (USD)</span>
            <input
              type="number"
              min={0.01}
              step={0.5}
              value={maxCostUsd}
              onChange={(e) => setMaxCostUsd(Number(e.target.value) || 5.0)}
            />
          </label>
        </div>
      )}

      <div className="form-actions">
        <button
          type="submit"
          className="btn-primary"
          disabled={!goal.trim() || isSubmitting}
        >
          <Play size={16} />
          {isSubmitting ? "Starting Workflow..." : "Run Workflow"}
        </button>
      </div>
    </form>
  );
}
