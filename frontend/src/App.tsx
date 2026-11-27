import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  History,
  LogOut,
  RefreshCw,
} from "lucide-react";
import {
  approveWorkflow,
  checkHealth,
  demoLogin,
  getAuthConfig,
  getCurrentUser,
  getWebSocketUrl,
  getWorkflowStatus,
  getWorkflowTrace,
  logout,
  rejectWorkflow,
  resumeWorkflow,
  startWorkflow,
  type AuthUser,
} from "./api/client";
import { ApprovalBanner } from "./components/ApprovalBanner";
import { BudgetPanel } from "./components/BudgetPanel";
import { LoginPage } from "./components/LoginPage";
import { PlanBoard } from "./components/PlanBoard";
import { TraceTimeline } from "./components/TraceTimeline";
import { WorkflowForm } from "./components/WorkflowForm";
import { WorkflowPipeline } from "./components/WorkflowPipeline";
import {
  normalizeStatus,
  type RunHistoryItem,
  type TraceEntry,
  type WorkflowRunRequest,
  type WorkflowStatusResponse,
} from "./types/workflow";

export function App() {
  const [authChecked, setAuthChecked] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [googleEnabled, setGoogleEnabled] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authBusy, setAuthBusy] = useState(false);

  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isActionBusy, setIsActionBusy] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const [history, setHistory] = useState<RunHistoryItem[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [statusData, setStatusData] = useState<WorkflowStatusResponse | null>(null);
  const [traceEntries, setTraceEntries] = useState<TraceEntry[]>([]);

  const wsRef = useRef<WebSocket | null>(null);

  // Check auth session on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const errParam = params.get("auth_error");
    if (errParam) {
      setAuthError(errParam);
      window.history.replaceState({}, "", window.location.pathname);
    }

    Promise.allSettled([getCurrentUser(), getAuthConfig()]).then(
      ([meRes, cfgRes]) => {
        if (meRes.status === "fulfilled" && meRes.value.authenticated) {
          setUser(meRes.value.user);
        }
        if (cfgRes.status === "fulfilled") {
          setGoogleEnabled(cfgRes.value.google_enabled);
        }
        setAuthChecked(true);
      }
    );
  }, []);

  const handleDemoLogin = async () => {
    setAuthBusy(true);
    setAuthError(null);
    try {
      const res = await demoLogin();
      setUser(res.user);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setAuthBusy(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } finally {
      setUser(null);
      setActiveRunId(null);
      setStatusData(null);
      setTraceEntries([]);
    }
  };

  // Check API health on mount
  const refreshHealth = useCallback(async () => {
    try {
      await checkHealth();
      setApiHealthy(true);
    } catch {
      setApiHealthy(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = setInterval(refreshHealth, 15000);
    return () => clearInterval(timer);
  }, [refreshHealth]);

  const fetchRunDetails = useCallback(async (runId: string) => {
    try {
      const [st, tr] = await Promise.all([
        getWorkflowStatus(runId),
        getWorkflowTrace(runId),
      ]);
      setStatusData(st);
      setTraceEntries(tr.entries);
      const norm = normalizeStatus(st.status);
      setHistory((prev) =>
        prev.map((item) =>
          item.run_id === runId ? { ...item, status: norm } : item
        )
      );
      return norm;
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to fetch run");
      return "unknown" as const;
    }
  }, []);

  // Poll + WebSocket for active run
  useEffect(() => {
    if (!activeRunId) return;

    let cancelled = false;
    fetchRunDetails(activeRunId);

    // Connect WebSocket for live push notifications
    try {
      const ws = new WebSocket(getWebSocketUrl(activeRunId));
      wsRef.current = ws;
      ws.onmessage = () => {
        if (!cancelled) {
          fetchRunDetails(activeRunId);
        }
      };
    } catch {
      // Fallback to polling below
    }

    const poll = setInterval(async () => {
      if (cancelled) return;
      const st = await fetchRunDetails(activeRunId);
      if (st === "completed" || st === "failed") {
        clearInterval(poll);
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(poll);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [activeRunId, fetchRunDetails]);

  const handleStartWorkflow = async (req: WorkflowRunRequest) => {
    setIsSubmitting(true);
    setErrorBanner(null);
    try {
      const res = await startWorkflow(req);
      const item: RunHistoryItem = {
        run_id: res.run_id,
        goal: req.goal,
        status: "running",
        created_at: new Date().toLocaleTimeString(),
        max_steps: req.max_steps,
        max_tokens: req.max_tokens,
        max_cost_usd: req.max_cost_usd,
      };
      setHistory((prev) => [item, ...prev]);
      setStatusData(null);
      setTraceEntries([]);
      setActiveRunId(res.run_id);
    } catch (err) {
      setErrorBanner(
        err instanceof Error ? err.message : "Failed to start workflow"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleApprove = async (feedback: string) => {
    if (!activeRunId) return;
    setIsActionBusy(true);
    setErrorBanner(null);
    try {
      await approveWorkflow(activeRunId, feedback);
      await fetchRunDetails(activeRunId);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Approve failed");
    } finally {
      setIsActionBusy(false);
    }
  };

  const handleReject = async (feedback: string) => {
    if (!activeRunId) return;
    setIsActionBusy(true);
    setErrorBanner(null);
    try {
      await rejectWorkflow(activeRunId, feedback);
      await fetchRunDetails(activeRunId);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Reject failed");
    } finally {
      setIsActionBusy(false);
    }
  };

  const handleResume = async () => {
    if (!activeRunId) return;
    setIsActionBusy(true);
    setErrorBanner(null);
    try {
      await resumeWorkflow(activeRunId);
      await fetchRunDetails(activeRunId);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Resume failed");
    } finally {
      setIsActionBusy(false);
    }
  };

  const activeHistoryItem = history.find((h) => h.run_id === activeRunId);
  const currentStatus = normalizeStatus(statusData?.status);

  if (!authChecked) {
    return (
      <div className="login-shell">
        <div className="muted">Checking session...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <LoginPage
        googleEnabled={googleEnabled}
        authError={authError}
        onDemoLogin={handleDemoLogin}
        isBusy={authBusy}
      />
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Bot size={22} className="icon-accent" />
          <div>
            <h1>Multi-Agent AI Workflow</h1>
            <span className="muted">LangGraph · Groq · MCP Tools · FastAPI</span>
          </div>
        </div>

        <div className="topbar-right">
          <div
            className={`health-pill ${
              apiHealthy === true
                ? "healthy"
                : apiHealthy === false
                  ? "offline"
                  : ""
            }`}
          >
            <Activity size={14} />
            <span>
              {apiHealthy === true
                ? "API Connected"
                : apiHealthy === false
                  ? "API Offline (port 8000)"
                  : "Checking API..."}
            </span>
          </div>
          <button
            type="button"
            className="btn-ghost btn-sm"
            onClick={() => activeRunId && fetchRunDetails(activeRunId)}
            disabled={!activeRunId}
            title="Refresh active run"
          >
            <RefreshCw size={14} />
          </button>
          <div className="user-pill">
            {user.picture ? (
              <img
                src={user.picture}
                alt={user.name}
                className="user-avatar"
                referrerPolicy="no-referrer"
              />
            ) : (
              <span className="user-avatar user-initials">
                {user.name.slice(0, 1).toUpperCase()}
              </span>
            )}
            <span className="user-name">{user.name}</span>
          </div>
          <button
            type="button"
            className="btn-ghost btn-sm"
            onClick={handleLogout}
            title="Log out"
          >
            <LogOut size={14} />
            Logout
          </button>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar card">
          <div className="card-header">
            <div className="card-title">
              <History size={16} />
              <h2>Runs</h2>
            </div>
            <span className="count-pill">{history.length}</span>
          </div>

          {history.length === 0 ? (
            <div className="empty-state">
              No runs yet. Submit a goal to start.
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <button
                  key={item.run_id}
                  type="button"
                  className={`history-item ${
                    item.run_id === activeRunId ? "selected" : ""
                  }`}
                  onClick={() => setActiveRunId(item.run_id)}
                >
                  <div className="history-top">
                    <span className={`status-pill status-${item.status}`}>
                      {item.status.replace("_", " ")}
                    </span>
                    <span className="muted">{item.created_at}</span>
                  </div>
                  <div className="history-goal">{item.goal}</div>
                </button>
              ))}
            </div>
          )}
        </aside>

        <main className="main-content">
          {errorBanner && (
            <div className="error-banner">
              <AlertCircle size={16} />
              <span>{errorBanner}</span>
            </div>
          )}

          <WorkflowForm
            onSubmit={handleStartWorkflow}
            isSubmitting={isSubmitting}
          />

          {activeRunId && statusData && (
            <>
              <div className="run-header card">
                <div>
                  <span className="muted">Run ID: {activeRunId}</span>
                  <h2 className="run-goal">{statusData.goal}</h2>
                </div>
                <span className={`status-pill status-${currentStatus}`}>
                  {currentStatus.replace("_", " ")}
                </span>
              </div>

              {statusData.error && (
                <div className="error-banner">
                  <AlertCircle size={16} />
                  <span>{statusData.error}</span>
                </div>
              )}

              {(currentStatus === "awaiting_approval" ||
                currentStatus === "paused") && (
                <ApprovalBanner
                  onApprove={handleApprove}
                  onReject={handleReject}
                  onResume={handleResume}
                  isBusy={isActionBusy}
                />
              )}

              <WorkflowPipeline
                status={currentStatus}
                plan={statusData.plan}
                currentTaskIndex={statusData.current_task_index}
                traceEntries={traceEntries}
              />

              <BudgetPanel
                stepCount={statusData.step_count}
                maxSteps={activeHistoryItem?.max_steps ?? 50}
                budget={statusData.budget}
                maxTokens={activeHistoryItem?.max_tokens ?? 100000}
                maxCostUsd={activeHistoryItem?.max_cost_usd ?? 5.0}
              />

              <PlanBoard
                plan={statusData.plan}
                currentTaskIndex={statusData.current_task_index}
                results={statusData.results}
              />

              <TraceTimeline entries={traceEntries} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
