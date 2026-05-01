# 🤖 AI Virtual Financial Advisor

A GenAI-powered autonomous agent for analyzing personal finance data, detecting financial risks, simulating what-if scenarios, and generating personalized natural language advice — all through an interactive Streamlit UI.

---

## ✨ Features

- 📊 **Multi-category transaction ingestion** — income & expense data from CSV
- 📈 **Financial health analysis** — savings rate, income vs. expense trends
- ⚠️ **Risky pattern detection** — overspending, idle savings, unusual large expenses
- 🔮 **What-if scenario simulation** — save 10% more, cut expenses 15%, or invest at 7% p.a.
- 💬 **GenAI-powered advice** — personalized natural language insights via Groq (`llama-3.3-70b-versatile`)
- 🧠 **Agentic reasoning loop** — autonomous multi-step analysis with session memory (`langchain_core`)
- 🖥️ **Streamlit UI** — interactive chat and dashboard
- 🐳 **Docker support** — container-ready deployment
- 🧪 **Auto test generation** — custom `TestGeneratorAgent` that writes pytest files from source AST

---

## 🗂️ Project Structure

```
virtual-financial-advisor/
├── data/
│   └── virtual_financial_advisor_data.csv   # Transaction dataset
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_expense_classification.ipynb
│   ├── 03_financial_trend_detection.ipynb
│   ├── 04_scenario_simulation.ipynb
│   ├── 05_genai_advice_generation.ipynb
│   └── 06_agent_loop_prototype.ipynb
├── src/
│   ├── data_loader.py           # CSV ingestion & preprocessing
│   ├── expense_classifier.py    # Category labelling & aggregation
│   ├── trend_detection.py       # Monthly trends & risky pattern detection
│   ├── scenario_simulation.py   # What-if financial simulations
│   ├── genai_interface.py       # Groq LLM wrapper (langchain_core)
│   └── agent/
│       ├── agent_core.py        # FinancialAdvisorSession — main agent loop
│       ├── tools.py             # LangChain-compatible tool definitions
│       ├── memory.py            # Session memory management
│       └── test_generator_agent.py  # Auto test-case generator agent
├── tests/                       # Pytest test suite (manual + auto-generated)
├── src/ui/
│   ├── cli.py                   # Command-line interface
│   └── streamlit_app.py         # Streamlit dashboard
├── Dockerfile
├── requirements.txt
├── AGENTS.md                    # Agent documentation
└── .env                         # API keys (not committed)
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-org/virtual-financial-advisor.git
cd virtual-financial-advisor
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com)

### 3. Run the Streamlit UI

```bash
streamlit run src/ui/streamlit_app.py
```

### 4. Run via CLI

```bash
python -m src.ui.cli
```

---

## 🧠 Agent Architecture

```
FinancialAdvisorSession (agent_core.py)
├── analyze()              ← Autonomous 5-step financial analysis
├── chat(query)            ← Interactive Q&A with memory
└── clear_history()        ← Reset session

Tools (tools.py)
├── analyze_financial_health   ← Income, expenses, savings rate
├── check_risky_patterns       ← Overspending, idle savings, outliers
├── simulate_saving_more       ← +10% savings for 12 months
├── simulate_expense_cut       ← −15% expenses for 12 months
└── simulate_investment        ← 7% p.a. growth over 10 years
```

### Python API

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

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run only auto-generated tests
pytest tests/ -v -k "generated"

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Auto Test Generation

The `TestGeneratorAgent` can generate pytest files from any source module:

```bash
# List registered modules
python -m src.agent.test_generator_agent --list

# Generate tests for a specific module
python -m src.agent.test_generator_agent --module src/trend_detection.py

# Generate tests for ALL registered modules
python -m src.agent.test_generator_agent --all

# Preview without writing
python -m src.agent.test_generator_agent --module src/expense_classifier.py --preview
```

---

## 🐳 Docker

```bash
# Build image
docker build -t virtual-financial-advisor .

# Run container
docker run -p 8501:8501 --env-file .env virtual-financial-advisor
```

---

## 🛠️ Tech Stack

| Layer          | Technology                                   |
|----------------|----------------------------------------------|
| Data           | Pandas, NumPy                                |
| LLM            | Groq (`llama-3.3-70b-versatile`), LangChain  |
| Agent Loop     | `langchain_core` (hand-rolled tool-call loop)|
| UI             | Streamlit, Plotly                            |
| Testing        | pytest, AST-based auto-generation            |
| Deployment     | Docker                                       |
| Config         | python-dotenv                                |

---

## 📄 License

MIT
