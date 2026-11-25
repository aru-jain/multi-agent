import {
  Brain,
  CheckCircle2,
  Code2,
  Database,
  GitBranch,
  Globe,
  ShieldCheck,
} from "lucide-react";
import type { SubTask, TraceEntry, WorkflowStatus } from "../types/workflow";

interface WorkflowPipelineProps {
  status: WorkflowStatus;
  plan: SubTask[];
  currentTaskIndex: number;
  traceEntries: TraceEntry[];
}

export function WorkflowPipeline({
  status,
  plan,
  currentTaskIndex,
  traceEntries,
}: WorkflowPipelineProps) {
  const lastAgent = traceEntries.length
    ? traceEntries[traceEntries.length - 1].agent?.toLowerCase()
    : undefined;

  const currentWorker = plan[currentTaskIndex]?.assigned_agent;

  const isPlannerDone = plan.length > 0;
  const isCompleted = status === "completed";

  const isActive = (node: string) => {
    if (isCompleted || status === "failed") return false;
    if (node === "planner" && !isPlannerDone && status === "running") return true;
    if (node === lastAgent) return true;
    if (node === currentWorker && status === "running") return true;
    return false;
  };

  return (
    <div className="card pipeline-card">
      <div className="pipeline-track">
        <div className={`pipeline-node ${status !== "unknown" ? "done" : ""}`}>
          <ShieldCheck size={16} />
          <span>Guard</span>
        </div>
        <span className="pipeline-arrow">→</span>

        <div
          className={`pipeline-node ${isPlannerDone ? "done" : ""} ${
            isActive("planner") ? "active" : ""
          }`}
        >
          <Brain size={16} />
          <span>Planner</span>
        </div>
        <span className="pipeline-arrow">→</span>

        <div
          className={`pipeline-node ${isPlannerDone ? "done" : ""} ${
            isActive("router") ? "active" : ""
          }`}
        >
          <GitBranch size={16} />
          <span>Router</span>
        </div>
        <span className="pipeline-arrow">→</span>

        <div className="pipeline-workers">
          <div
            className={`pipeline-node worker-researcher ${
              isActive("researcher") ? "active" : ""
            }`}
          >
            <Globe size={15} />
            <span>Researcher</span>
          </div>
          <div
            className={`pipeline-node worker-coder ${
              isActive("coder") ? "active" : ""
            }`}
          >
            <Code2 size={15} />
            <span>Coder</span>
          </div>
          <div
            className={`pipeline-node worker-db ${
              isActive("db_analyst") ? "active" : ""
            } ${status === "awaiting_approval" ? "approval" : ""}`}
          >
            <Database size={15} />
            <span>DB Analyst</span>
          </div>
        </div>
        <span className="pipeline-arrow">→</span>

        <div
          className={`pipeline-node ${isCompleted ? "done" : ""} ${
            isActive("evaluator") ? "active" : ""
          }`}
        >
          <CheckCircle2 size={16} />
          <span>Evaluator</span>
        </div>
      </div>
    </div>
  );
}
