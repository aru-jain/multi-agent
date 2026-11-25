"""Evaluation scenarios definition for the Multi-Agent AI Workflow system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EvalScenario",
    "SCENARIOS",
    "get_scenario",
    "get_scenarios_by_difficulty",
]


@dataclass
class EvalScenario:
    """Definition of a single evaluation scenario.

    Attributes:
        id: Unique identifier for the scenario.
        name: Short human-readable title.
        goal: Target objective or prompt that initiates the workflow.
        difficulty: Difficulty level ('easy', 'medium', 'hard', 'stress').
        expected_agents: List of agent roles expected to participate.
        expected_tools: List of tool names expected to be invoked.
        success_criteria: Description of what constitutes successful completion.
        max_steps_override: Optional override for the maximum workflow step limit.
        max_tokens_override: Optional override for the maximum workflow token limit.
        inject_failure: Optional failure condition to inject for error testing.
    """

    id: int
    name: str
    goal: str
    difficulty: str  # "easy", "medium", "hard", "stress"
    expected_agents: list[str]  # Which agents should be involved
    expected_tools: list[str]  # Which tools should be called
    success_criteria: str  # Description of what constitutes success
    max_steps_override: int | None = None  # Override max steps for stress tests
    max_tokens_override: int | None = None
    inject_failure: str | None = None  # For testing error handling


SCENARIOS: list[EvalScenario] = [
    EvalScenario(
        id=1,
        name="Simple factual query",
        goal="What is the capital of France? Provide a brief answer.",
        difficulty="easy",
        expected_agents=["researcher"],
        expected_tools=["web_search"],
        success_criteria="Response mentions Paris",
    ),
    EvalScenario(
        id=2,
        name="Weather lookup",
        goal="Find the current weather conditions in Tokyo, Japan.",
        difficulty="easy",
        expected_agents=["researcher"],
        expected_tools=["web_search", "fetch_url"],
        success_criteria="Response includes temperature or weather conditions for Tokyo",
    ),
    EvalScenario(
        id=3,
        name="Simple computation",
        goal="Calculate the factorial of 20 using Python code.",
        difficulty="easy",
        expected_agents=["coder"],
        expected_tools=["execute_python"],
        success_criteria="Response includes 2432902008176640000",
    ),
    EvalScenario(
        id=4,
        name="Code linting",
        goal="Lint the following Python code and fix any errors: def hello(name)\n  print(f'Hello {name}')",
        difficulty="easy",
        expected_agents=["coder"],
        expected_tools=["lint_code", "execute_python"],
        success_criteria="Response provides corrected code with proper indentation",
    ),
    EvalScenario(
        id=5,
        name="Top frameworks research",
        goal="Find the top 5 Python web frameworks and briefly describe each.",
        difficulty="easy",
        expected_agents=["researcher"],
        expected_tools=["web_search"],
        success_criteria="Response lists at least 5 frameworks including Django and Flask",
    ),
    EvalScenario(
        id=6,
        name="Data visualization",
        goal="Write Python code to create a list of sales data (Widget A: 1250, Widget B: 890, Widget C: 2100, Gadget X: 3450, Gadget Y: 1800) and calculate the total and average sales.",
        difficulty="medium",
        expected_agents=["coder"],
        expected_tools=["execute_python"],
        success_criteria="Response includes correct total (9490) and average (1898)",
    ),
    EvalScenario(
        id=7,
        name="News summarization",
        goal="Research and summarize the latest developments in artificial intelligence from the past week.",
        difficulty="medium",
        expected_agents=["researcher"],
        expected_tools=["web_search", "fetch_url"],
        success_criteria="Response provides a structured summary with multiple AI-related topics",
    ),
    EvalScenario(
        id=8,
        name="Database query",
        goal="Query the database to find all items in the 'sales' category from the sample_data table and report the total value.",
        difficulty="medium",
        expected_agents=["db_analyst"],
        expected_tools=["query_db"],
        success_criteria="Response includes correct sales data from the database",
    ),
    EvalScenario(
        id=9,
        name="CSV parsing script",
        goal="Write a Python script that parses this CSV data and computes statistics: 'name,age,score\nAlice,25,88\nBob,30,92\nCharlie,28,85\nDiana,22,95'. Calculate mean age and mean score.",
        difficulty="medium",
        expected_agents=["coder"],
        expected_tools=["execute_python"],
        success_criteria="Response includes correct mean age (26.25) and mean score (90.0)",
    ),
    EvalScenario(
        id=10,
        name="Research then code",
        goal="Research what the Fibonacci sequence is, then write Python code to generate the first 20 Fibonacci numbers.",
        difficulty="medium",
        expected_agents=["researcher", "coder"],
        expected_tools=["web_search", "execute_python"],
        success_criteria="Response includes explanation of Fibonacci and correct first 20 numbers",
    ),
    EvalScenario(
        id=11,
        name="Research and store",
        goal="Research the populations of the 5 largest cities in the world and store the results in the database sample_data table with category 'population'.",
        difficulty="medium",
        expected_agents=["researcher", "db_analyst"],
        expected_tools=["web_search", "write_db"],
        success_criteria="Research completed and data insertion attempted (may require approval)",
    ),
    EvalScenario(
        id=12,
        name="DB analysis pipeline",
        goal="Query all data from the sample_data table, then write Python code to analyze it: calculate totals per category and find the highest-value item.",
        difficulty="hard",
        expected_agents=["db_analyst", "coder"],
        expected_tools=["query_db", "execute_python"],
        success_criteria="Response includes per-category analysis and identifies the highest-value item",
    ),
    EvalScenario(
        id=13,
        name="Debug failing code",
        goal="Debug this Python function that should return the sum of even numbers but has bugs: def sum_evens(n):\n    total = 0\n    for i in range(n):\n        if i % 2 = 0:\n            total += i\n    return total\nTest it with n=10, expected result is 20.",
        difficulty="hard",
        expected_agents=["coder"],
        expected_tools=["execute_python", "lint_code"],
        success_criteria="Response identifies the bug (= vs ==), fixes it, and verifies output is 20",
    ),
    EvalScenario(
        id=14,
        name="Full pipeline",
        goal="Research the current exchange rates for USD to EUR, GBP, and JPY. Then write Python code to calculate conversions for $100, $500, and $1000. Finally, store the results in the database.",
        difficulty="hard",
        expected_agents=["researcher", "coder", "db_analyst"],
        expected_tools=["web_search", "execute_python", "write_db"],
        success_criteria="Exchange rates researched, conversions calculated, and DB write attempted",
    ),
    EvalScenario(
        id=15,
        name="ETL pipeline",
        goal="Fetch data from https://jsonplaceholder.typicode.com/users (first 5 users), transform it with Python to extract just name and email, then store in the database.",
        difficulty="hard",
        expected_agents=["researcher", "coder", "db_analyst"],
        expected_tools=["fetch_url", "execute_python", "write_db"],
        success_criteria="User data fetched, transformed, and DB insertion attempted",
    ),
    EvalScenario(
        id=16,
        name="Multi-step analysis with recovery",
        goal="Query the sample_data table for all records, then write Python code to find statistical outliers (values more than 2 standard deviations from mean). If no outliers found, explain why.",
        difficulty="hard",
        expected_agents=["db_analyst", "coder"],
        expected_tools=["query_db", "execute_python"],
        success_criteria="Response includes statistical analysis with mean, std dev, and outlier determination",
    ),
    EvalScenario(
        id=17,
        name="Tool failure recovery",
        goal="Search the web for 'latest Python release version', then write code to print that version number. Note: if search fails, use your knowledge to provide the answer.",
        difficulty="hard",
        expected_agents=["researcher", "coder"],
        expected_tools=["web_search", "execute_python"],
        success_criteria="Response provides a Python version number despite potential tool failures",
        inject_failure="web_search",
    ),
    EvalScenario(
        id=18,
        name="Replan after failure",
        goal="Try to query a non-existent table 'nonexistent_table' in the database. When that fails, replan to query the 'sample_data' table instead and report what you find.",
        difficulty="hard",
        expected_agents=["db_analyst"],
        expected_tools=["query_db"],
        success_criteria="System recovers from the initial failure and successfully queries sample_data",
    ),
    EvalScenario(
        id=19,
        name="Step limit stress test",
        goal="Write a very complex Python program that generates and analyzes a dataset of 1000 random numbers: compute mean, median, mode, standard deviation, percentiles, and create a frequency distribution.",
        difficulty="stress",
        expected_agents=["coder"],
        expected_tools=["execute_python"],
        success_criteria="Task completes within step limit or gracefully terminates with guard message",
        max_steps_override=15,
    ),
    EvalScenario(
        id=20,
        name="Budget limit stress test",
        goal="Research 10 different programming languages, their creators, year of creation, and key features. Then write Python code to create a comparison table.",
        difficulty="stress",
        expected_agents=["researcher", "coder"],
        expected_tools=["web_search", "execute_python"],
        success_criteria="Task hits budget limit and gracefully terminates with guard message",
        max_tokens_override=5000,
    ),
]


def get_scenario(scenario_id: int) -> EvalScenario | None:
    """Get an evaluation scenario by its unique identifier.

    Args:
        scenario_id: The integer identifier of the scenario to retrieve.

    Returns:
        The matching EvalScenario instance, or None if not found.
    """
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None


def get_scenarios_by_difficulty(difficulty: str) -> list[EvalScenario]:
    """Get all evaluation scenarios matching a given difficulty level.

    Args:
        difficulty: Difficulty category ("easy", "medium", "hard", "stress").

    Returns:
        A list of EvalScenario instances matching the specified difficulty.
    """
    return [s for s in SCENARIOS if s.difficulty == difficulty]
