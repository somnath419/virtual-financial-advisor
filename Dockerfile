FROM python:3.11-slim

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── install Python deps first (layer-cached unless requirements change) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── make src importable from any working directory ───────────────────────────
ENV PYTHONPATH=/app

# ── tell the app where the data lives (can be overridden at runtime) ──────────
ENV DATA_PATH=/app/data/virtual_financial_advisor_data.csv

# ── Streamlit port ────────────────────────────────────────────────────────────
EXPOSE 8501

# ── default: launch the Streamlit UI ─────────────────────────────────────────
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
