# Pro Trading Terminal 📈

A professional-grade trading dashboard built with **Streamlit**, featuring real-time market data, 100+ technical analysis features, and AI-powered insights.

## Quickstart (Local)

```bash
./start.sh
```

This creates a virtual environment, installs dependencies, and starts the app.

**Manual steps** (if you prefer):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Features

| Tab | Highlights |
|---|---|
| 🕯️ **Chart** | Candlestick / Heikin-Ashi, EMA, RSI, MACD, Bollinger, Ichimoku, drawing tools |
| 🏢 **Fundamental** | Company profile, analyst recs, financial statements, ESG, sentiment radar |
| 🎲 **Quants** | VaR, CVaR, Monte Carlo simulation, Sharpe/Sortino/Calmar, Kelly Criterion |
| 💼 **Portfolio** | Paper trading, allocation charts, seasonality, screeners, watchlist |
| 🎯 **Strategies** | Bollinger Scalping, Reversal, Fibonacci Swing with entry/exit signals |
| 🤖 **AI & ML** | Pattern scanner, regime detection, auto S/R, price forecast, risk profiler |

## Tech Stack

- **Frontend:** Streamlit + Plotly interactive charts
- **Data:** yfinance (real-time market data)
- **Analytics:** NumPy, SciPy, Pandas
- **Testing:** pytest
- **CI/CD:** GitHub Actions
- **Deployment:** Docker

## Docker Deployment

```bash
docker-compose up --build -d
```

The app is then available at [http://localhost:8501](http://localhost:8501).

## Testing

```bash
pytest test_app.py -v
```

## Multi-Agent Development

See [AGENTS.md](AGENTS.md) for the multi-agent orchestration system governing collaborative development on this project.
