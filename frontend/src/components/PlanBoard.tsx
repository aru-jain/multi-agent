import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Code2,
  Database,
  Globe,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import type { AgentRole, SubTask } from "../types/workflow";

interface PlanBoardProps {
  plan: SubTask[];
  currentTaskIndex: number;
  results: Record<string, unknown>;
}

function AgentBadge({ role }: { role: AgentRole }) {
  const config: Record<string, { label: string; icon: JSX.Element; cls: string }> = {
    researcher: {
      label: "Researcher",
      icon: <Globe size={12} />,
      cls: "badge-researcher",
    },
    coder: {
      label: "Coder",
      icon: <Code2 size={12} />,
      cls: "badge-coder",
    },
    db_analyst: {
      label: "DB Analyst",
      icon: <Database size={12} />,
      cls: "badge-db",
    },
  };
  const info = config[role] || {
    label: role,
    icon: <Clock size={12} />,
    cls: "badge-neutral",
  };
  return (
    <span className={`badge ${info.cls}`}>
      {info.icon}
      {info.label}
    </span>
  );
}

export function PlanBoard({ plan, currentTaskIndex, results }: PlanBoardProps) {
  return (
    <div className="card">
      <div className="card-header">
        <h2>Execution Plan</h2>
        <span className="muted">
          {plan.length > 0
            ? `${Math.min(currentTaskIndex, plan.length)} / ${plan.length} subtasks`
            : "Waiting for planner..."}
        </span>
      </div>

      {plan.length === 0 ? (
        <div className="empty-state">
          Planner is decomposing the goal into ordered subtasks...
        </div>
      ) : (
        <div className="task-list">
          {plan.map((task, idx) => {
            const isCurrent = idx === currentTaskIndex;
            const taskStatus = String(task.status || "pending")
              .toLowerCase()
              .replace("taskstatus.", "");
            const taskOutput = task.result ?? results[task.id];

            return (
              <div
                key={task.id || idx}
                className={`task-item ${isCurrent ? "task-current" : ""} task-${taskStatus}`}
              >
                <div className="task-top">
                  <div className="task-id-group">
                    <span className="task-index">#{idx + 1}</span>
                    <AgentBadge role={task.assigned_agent} />
                    {task.requires_approval && (
                      <span className="badge badge-warning">
                        <ShieldAlert size={12} />
                        Approval Gate
                      </span>
                    )}
                  </div>
                  <span className={`status-pill status-${taskStatus}`}>
                    {taskStatus === "completed" && <CheckCircle2 size={12} />}
                    {taskStatus === "in_progress" && (
                      <Loader2 size={12} className="spin" />
                    )}
                    {taskStatus === "failed" && <AlertTriangle size={12} />}
                    {taskStatus.replace("_", " ")}
                  </span>
                </div>

                <p className="task-desc">{task.description}</p>

                {task.dependencies && task.dependencies.length > 0 && (
                  <div className="task-deps">
                    Depends on: {task.dependencies.join(", ")}
                  </div>
                )}

                {taskOutput !== undefined && taskOutput !== null && (
                  <pre className="task-result">
                    {typeof taskOutput === "string"
                      ? taskOutput
                      : JSON.stringify(taskOutput, null, 2)}
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
