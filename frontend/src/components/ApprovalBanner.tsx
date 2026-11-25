import { useState } from "react";
import { Check, Play, ShieldAlert, X } from "lucide-react";

interface ApprovalBannerProps {
  onApprove: (feedback: string) => Promise<void>;
  onReject: (feedback: string) => Promise<void>;
  onResume: () => Promise<void>;
  isBusy: boolean;
}

export function ApprovalBanner({
  onApprove,
  onReject,
  onResume,
  isBusy,
}: ApprovalBannerProps) {
  const [feedback, setFeedback] = useState("");

  return (
    <div className="card approval-banner">
      <div className="approval-header">
        <ShieldAlert size={20} className="icon-warning" />
        <div>
          <h3>Human Approval Checkpoint</h3>
          <p className="muted">
            The workflow paused before executing a database tool (`db_tools`).
            Review the trace and approve or reject with feedback.
          </p>
        </div>
      </div>

      <div className="approval-controls">
        <input
          type="text"
          className="approval-input"
          placeholder="Optional feedback or rejection reason..."
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          disabled={isBusy}
        />
        <div className="approval-buttons">
          <button
            type="button"
            className="btn-success"
            disabled={isBusy}
            onClick={() => onApprove(feedback)}
          >
            <Check size={15} />
            Approve
          </button>
          <button
            type="button"
            className="btn-danger"
            disabled={isBusy}
            onClick={() => onReject(feedback)}
          >
            <X size={15} />
            Reject
          </button>
          <button
            type="button"
            className="btn-ghost"
            disabled={isBusy}
            onClick={onResume}
          >
            <Play size={15} />
            Resume
          </button>
        </div>
      </div>
    </div>
  );
}
