# Wealth Management Agent MVP

This is an automated wealth management platform MVP that uses Agentic Workflows to process real-time news, decide investment actions, and simulate execution.

## Architecture

1. **Data Ingestion Layer** (`ingestion.py`): Scrapes or simulates real-time financial news using BeautifulSoup/requests.
2. **Analysis Layer / Intelligence** (`intelligence.py`): LangGraph workflow with cycles:
   - **Analyst Agent**: Extracts Sentiment, Affected Assets, and Confidence using GPT-4o with structured Pydantic outputs.
   - **Strategist Agent**: Evaluates the output and generates a signal (BUY/HOLD/SELL) and a target asset.
   - **Risk Agent**: Validates the Strategist's signal. If confidence is too low or sentiment contradicts the signal, it sends it back to the Strategist for revision (a cyclic graph).
3. **Market Data Layer** (`market_data.py`): Uses `yfinance` to fetch the current asset price.
4. **Execution Simulator** (`execution.py`): Mocks a portfolio, calculates position sizing based on risk (e.g. 5%), logs the trade, and updates the balance.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Google API Key:
   Create a `.env` file in the root directory and add:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the MVP:
   - **CLI Mode**:
     ```bash
     python main.py
     ```
   - **Dashboard Mode (Recommended)**:
     ```bash
     streamlit run app.py
     ```
   - **Autonomous Bot Mode (Always On)**:
     ```bash
     # Run in a separate terminal or screen/tmux
     python3 trading_bot.py
     ```

## Configuration (Optional)
In your `.env` file, you can adjust:
- `TRADING_INTERVAL_SECONDS`: Frequency of market scans (default: 600)
- `TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID`: For mobile notifications.

## Project Structure

```text
wealth_agent_mvp/
├── app.py
├── main.py
├── trading_bot.py
├── train_sentinel.py
├── backtest_engine.py
├── *.py (core modules)
├── tests/                # utility test/check scripts
├── data/                 # generated CSV datasets and backtest outputs
├── artifacts/            # trained ML model artifacts (.pkl)
├── storage/              # SQLite databases
├── logs/                 # runtime and training logs
└── paths.py              # centralized file paths
```
