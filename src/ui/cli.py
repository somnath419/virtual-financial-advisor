"""
Virtual Financial Advisor – CLI interface
==========================================
Interactive command-line loop backed by FinancialAdvisorSession.

The agent follows the THINK → ACT → OBSERVE → REASON → RESPOND loop,
autonomously calling the right tools and maintaining full session memory.

Usage (from project root):
    python -m src.ui.cli

Special commands:
    analyze   – trigger a full autonomous 5-step financial analysis
    history   – print the conversation history for this session
    clear     – reset the session memory
    exit      – quit
"""

import os
import sys

# Ensure project root is on PYTHONPATH when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.agent.agent_core import create_session  # noqa: E402


def main() -> None:
    """
    Run the interactive CLI loop backed by FinancialAdvisorSession.

    Prompts for a user ID, then enters a question-answer loop.
    The session persists conversation memory across all questions.
    Type ``analyze`` to trigger a full autonomous analysis without prompting.
    """
    print("╔══════════════════════════════════════════════╗")
    print("║   💰  Virtual Financial Advisor  – CLI       ║")
    print("╚══════════════════════════════════════════════╝")
    print("Commands:  analyze | history | clear | exit\n")

    user_id = input("Enter User ID (e.g. user_1): ").strip()
    if not user_id:
        user_id = "user_1"

    print(f"\n✅  Session started for '{user_id}'.  Type 'analyze' for a full report.\n")
    session = create_session(user_id)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        # ── special commands ───────────────────────────────────────────────
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if question.lower() == "analyze":
            print("\n🔍  Running autonomous 5-step analysis…\n")
            answer = session.analyze()
            print(f"\nAdvisor:\n{answer}\n")
            continue

        if question.lower() == "history":
            if not session.history:
                print("  (no conversation history yet)\n")
            for turn in session.history:
                role = "You" if turn["role"] == "user" else "Advisor"
                print(f"  [{role}]: {turn['content'][:120]}…\n")
            continue

        if question.lower() == "clear":
            session.clear_history()
            print("  ✅  Session memory cleared.\n")
            continue

        # ── normal question ────────────────────────────────────────────────
        try:
            answer = session.chat(question)
            print(f"\nAdvisor: {answer}\n")
        except Exception as exc:
            print(f"\n⚠️  Error: {exc}\n")


if __name__ == "__main__":
    main()
