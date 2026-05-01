"""
agent_core.py – Agentic Reasoning Loop
========================================
Dependency-stable design: only langchain_core is required.
No AgentExecutor, no ConversationBufferMemory, no internal langchain paths.

Public API
----------
create_session(user_id)  → FinancialAdvisorSession   (preferred)
create_agent()           → FinancialAdvisorSession   (backwards-compatible)

FinancialAdvisorSession
  .chat(question)   – interactive turn; agent calls tools autonomously
  .analyze()        – full autonomous 5-step analysis loop
  .invoke(inputs)   – AgentExecutor-compatible shim  {input:} → {output:}
  .clear_history()  – reset messages + display history
  .turn_count       – number of completed conversation turns
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.agent.tools import (
    analyze_financial_health,
    check_risky_patterns,
    simulate_expense_cut,
    simulate_investment,
    simulate_saving_more,
)
from src.data_loader import load_data
from src.genai_interface import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Shared DataFrame – loaded once; all tools import `df` from here
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "data", "virtual_financial_advisor_data.csv",
    ),
)
df = load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------
_TOOLS: List[Any] = [
    analyze_financial_health,
    check_risky_patterns,
    simulate_saving_more,
    simulate_expense_cut,
    simulate_investment,
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a professional virtual financial advisor with access to real "
    "financial data tools.\n\n"
    "Reasoning loop you MUST follow for every request:\n"
    "  1. THINK   – What data do I need? Which tools provide it?\n"
    "  2. ACT     – Call the tools in a logical sequence.\n"
    "  3. OBSERVE – What do the results reveal?\n"
    "  4. REASON  – Are there patterns across multiple tool outputs?\n"
    "  5. RESPOND – Give specific, data-backed advice with concrete numbers.\n\n"
    "Rules:\n"
    "• Always fetch real data via tools BEFORE drawing conclusions.\n"
    "• If a full analysis is requested, call ALL relevant tools autonomously "
    "– do NOT wait for prompts between steps.\n"
    "• Build on prior conversation context instead of repeating earlier findings.\n"
    "• Prioritise recommendations by impact (highest first).\n"
    "• Include dollar amounts and percentages in every recommendation."
)


# ---------------------------------------------------------------------------
# FinancialAdvisorSession
# ---------------------------------------------------------------------------

class FinancialAdvisorSession:
    """
    Stateful agentic reasoning session backed by a hand-rolled tool-call loop.

    Uses only ``langchain_core`` — works with every LangChain version that
    supports ``llm.bind_tools()``.  No AgentExecutor, no ConversationBufferMemory.

    Attributes:
        user_id (str): User being analysed in this session.
        history (list): Display-friendly conversation history.

    Example::

        session = create_session("user_1")
        report  = session.analyze()
        answer  = session.chat("What is my biggest risk?")
        answer  = session.chat("How does that compare to last month?")
        session.clear_history()
    """

    _MAX_ITERATIONS: int = 12

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._llm = get_llm().bind_tools(_TOOLS)
        self._tool_map: Dict[str, Any] = {t.name: t for t in _TOOLS}
        self._messages: List[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
        self.history: List[Dict[str, str]] = []

    # ── core reasoning loop ────────────────────────────────────────────────

    def chat(self, question: str) -> str:
        """
        Send *question* through the THINK → ACT → OBSERVE → REASON → RESPOND loop.

        Args:
            question (str): User's question or instruction.

        Returns:
            str: Data-backed answer from the agent.
        """
        enriched = f"[user_id: {self.user_id}]  {question}"
        self._messages.append(HumanMessage(content=enriched))

        answer = ""
        try:
            for _ in range(self._MAX_ITERATIONS):
                response: AIMessage = self._llm.invoke(self._messages)
                self._messages.append(response)

                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    answer = response.content or ""
                    break

                # Execute every requested tool call
                for tc in tool_calls:
                    tool_name: str = tc.get("name", "")
                    tool_args: dict = tc.get("args", {})
                    tool_id: str   = tc.get("id", tool_name)

                    if tool_name in self._tool_map:
                        try:
                            result = self._tool_map[tool_name].invoke(tool_args)
                        except Exception as exc:
                            result = f"Tool '{tool_name}' error: {exc}"
                    else:
                        result = f"Unknown tool: '{tool_name}'"

                    self._messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id)
                    )
            else:
                last = self._messages[-1]
                answer = getattr(last, "content", "Analysis reached max iterations.")

        except Exception as exc:
            answer = f"⚠️  Agent error: {exc}"

        self.history.append({"role": "user",     "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    # ── high-level helpers ─────────────────────────────────────────────────

    def analyze(self) -> str:
        """
        Trigger a complete autonomous 5-step financial analysis.

        Steps executed without user prompting:
          1. analyze_financial_health  – income, expenses, savings rate
          2. check_risky_patterns      – overspending, idle savings, outliers
          3. simulate_saving_more      – +10 % savings × 12 months
          4. simulate_expense_cut      – −15 % expenses × 12 months
          5. simulate_investment       – 7 % p.a. × 10 years
          6. 3 prioritised dollar-specific recommendations
        """
        prompt = (
            f"Perform a complete autonomous financial analysis for user "
            f"'{self.user_id}'.\n\n"
            "Execute every step – do NOT skip any and do NOT ask for confirmation:\n\n"
            "Step 1 – Call `analyze_financial_health`  → income, expenses, savings rate.\n"
            "Step 2 – Call `check_risky_patterns`      → all risk findings with severity.\n"
            "Step 3 – Call `simulate_saving_more`      → impact of +10 % savings × 12 mo.\n"
            "Step 4 – Call `simulate_expense_cut`      → impact of −15 % expenses × 12 mo.\n"
            "Step 5 – Call `simulate_investment`       → portfolio at 7 % p.a. × 10 yr.\n"
            "Step 6 – Give exactly 3 prioritised recommendations with specific "
            "dollar amounts, ordered by highest financial impact."
        )
        return self.chat(prompt)

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """
        AgentExecutor-compatible shim.

        ``agent.invoke({"input": "..."})["output"]`` works unchanged.
        """
        return {"output": self.chat(inputs.get("input", ""))}

    def clear_history(self) -> None:
        """Reset display history and LLM message list to a clean slate."""
        self.history.clear()
        self._messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    @property
    def turn_count(self) -> int:
        """Number of completed conversation turns."""
        return len(self.history) // 2


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def create_session(user_id: str) -> FinancialAdvisorSession:
    """
    Create a ``FinancialAdvisorSession`` scoped to *user_id*.

    Example::

        session = create_session("user_1")
        print(session.analyze())
    """
    return FinancialAdvisorSession(user_id=user_id)


def create_agent(user_id: str = "default") -> FinancialAdvisorSession:
    """
    Backwards-compatible factory.

    Returns a ``FinancialAdvisorSession`` that also supports
    ``agent.invoke({"input": "..."})`` so existing notebooks work unchanged.
    """
    return FinancialAdvisorSession(user_id=user_id)
