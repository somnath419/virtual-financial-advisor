# AGENTS.md — Virtual Financial Advisor

This file documents the custom agents available in this project.

---

## 🤖 TestGeneratorAgent

**Location:** `src/agent/test_generator_agent.py`

### Purpose
An autonomous test-case generator agent that:
1. **Analyzes** any Python source module using static AST parsing (no imports required).
2. **Builds** a structured prompt describing every public function's signature, arguments, and docstring.
3. **Calls the LLM** (Groq `llama-3.3-70b-versatile` via `src/genai_interface.py`) to write comprehensive pytest test cases.
4. **Writes** the generated test file to `tests/test_<module>_generated.py`.
5. **Falls back** to stub-only generation if the LLM is unavailable (quota/key issues).

### Registered Modules

| Key                  | Source File                     |
|----------------------|---------------------------------|
| `data_loader`        | `src/data_loader.py`            |
| `expense_classifier` | `src/expense_classifier.py`     |
| `trend_detection`    | `src/trend_detection.py`        |
| `scenario_simulation`| `src/scenario_simulation.py`    |
| `genai_interface`    | `src/genai_interface.py`        |
| `agent_tools`        | `src/agent/tools.py`            |
| `agent_core`         | `src/agent/agent_core.py`       |

### CLI Usage

```bash
# List all registered modules
python -m src.agent.test_generator_agent --list

# Generate tests for a specific module
python -m src.agent.test_generator_agent --module src/trend_detection.py

# Generate tests and place them in a custom output directory
python -m src.agent.test_generator_agent --module src/scenario_simulation.py --output tests/

# Preview generated code without writing to disk
python -m src.agent.test_generator_agent --module src/expense_classifier.py --preview

# Generate tests for ALL registered modules at once
python -m src.agent.test_generator_agent --all
```

### Python API

```python
from src.agent.test_generator_agent import TestGeneratorAgent

# Generate tests and write to disk
agent = TestGeneratorAgent("src/trend_detection.py", output_dir="tests")
output_path = agent.generate()        # → tests/test_trend_detection_generated.py

# Preview without writing
code = agent.preview()
print(code)
```

### Pre-generated Test Files

The following test files were pre-generated and are ready to run:

| Test File                                    | Module Tested                |
|----------------------------------------------|------------------------------|
| `tests/test_data_loader_generated.py`        | `src/data_loader.py`         |
| `tests/test_expense_classifier_generated.py` | `src/expense_classifier.py`  |
| `tests/test_trend_detection_generated.py`    | `src/trend_detection.py`     |
| `tests/test_scenario_simulation_generated.py`| `src/scenario_simulation.py` |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only generated tests
pytest tests/ -v -k "generated"

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Architecture

```
TestGeneratorAgent
├── extract_functions(source_path)   ← AST parser → List[FunctionInfo]
├── _build_prompt()                  ← Structured LLM prompt
├── _call_llm(prompt)                ← Groq LLM via langchain_core
│   └── _template_fallback()        ← Stub generation (no LLM)
├── preview()                        ← Returns code string
└── generate()                       ← Writes test file to disk
```

---

## 🤖 FinancialAdvisorSession (Main Agent)

**Location:** `src/agent/agent_core.py`

An autonomous financial analysis agent backed by a hand-rolled tool-call loop using `langchain_core`.

### Tools

| Tool                       | Description                                         |
|----------------------------|-----------------------------------------------------|
| `analyze_financial_health` | Income, expenses, savings rate                      |
| `check_risky_patterns`     | Overspending, idle savings, outlier expenses        |
| `simulate_saving_more`     | +10% savings for 12 months                         |
| `simulate_expense_cut`     | −15% expenses for 12 months                        |
| `simulate_investment`      | 7% p.a. portfolio growth for 10 years              |

### Usage

```python
from src.agent.agent_core import create_session

session = create_session("user_1")

# Full autonomous 5-step analysis
report = session.analyze()
print(report)

# Interactive Q&A with session memory
answer = session.chat("What is my biggest financial risk?")
answer = session.chat("How much would I save by cutting dining expenses by 20%?")

# Reset memory
session.clear_history()
```

