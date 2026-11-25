"""Agent definitions and node factories for the Multi-Agent AI Workflow system."""

from __future__ import annotations

from src.agents.coder import CODER_SYSTEM_PROMPT, create_coder_node
from src.agents.db_analyst import DB_ANALYST_SYSTEM_PROMPT, create_db_analyst_node
from src.agents.evaluator import EVALUATOR_SYSTEM_PROMPT, evaluator_node
from src.agents.planner import PLANNER_SYSTEM_PROMPT, planner_node
from src.agents.researcher import RESEARCHER_SYSTEM_PROMPT, create_researcher_node

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "planner_node",
    "RESEARCHER_SYSTEM_PROMPT",
    "create_researcher_node",
    "CODER_SYSTEM_PROMPT",
    "create_coder_node",
    "DB_ANALYST_SYSTEM_PROMPT",
    "create_db_analyst_node",
    "EVALUATOR_SYSTEM_PROMPT",
    "evaluator_node",
]
