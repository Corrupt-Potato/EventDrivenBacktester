# Python Event-Driven Backtester

A modular, high-fidelity event-driven backtesting engine in Python — designed for realistic portfolio simulation using Alpaca’s historical API.

Supports fractional shares, end-of-bar logic, benchmark comparison, and plug-and-play strategy functions.

---

## 🚀 Features

- ✅ Event-driven architecture with clean separation of logic  
- 🔁 End-of-bar execution for realistic trade simulation  
- 📊 Fractional share support (float precision)  
- 📦 Alpaca integration with Parquet caching  
- 💸 Portfolio manager with position tracking, PnL, and equity curves  
- 📈 Benchmark support (auto-allocation or custom weights)  
- 🔧 Strategy plug-in via Python functions  
- ⚙️ Realistic trading constraints (cash checks, slippage, share limits)

---

## 🛠 Tech Stack

- **Python** 3.8+  
- `pandas`
- `numpy`
- `pyarrow` (Parquet)  
- `alpaca-py` (historical data)  
- `matplotlib` (plotting)  
- `pytz` (tz management)

---

## 📦 Installation & Usage

1. **Clone** this repo  
   ```bash
   git clone https://github.com/Corrupt-Potato/backtester.git
   cd backtester
   ```

2. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```

> ⚠️ Currently only supports **Alpaca API** input.  
> 📁 CSV support is coming soon!

---

## 📈 Sample Equity Curve

Below is a sample daily rebalancing backtest (2017–2024) of an equal-weight strategy (AAPL, MSFT, GOOG, AMZN, NVDA, TSLA, META) vs. SPY benchmark.

![Equity Curve](Figure_1.png)

---

## 🧑‍💻 Backtester Example Code

```python
import pandas as pd
from EventBacktester import Backtester

print("--- Running Backtester Example (Float Mode) ---")

# 1. Define Strategy Logic (returns float weights)
def simple_equal_weight_strategy(timestamp, current_data, portfolio_state):
    strategy_tickers_in_use = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']
    
    available_tickers = [t for t in strategy_tickers_in_use if t in current_data.columns and not pd.isna(current_data[t].iloc[-1])]
    if not available_tickers: return {}
    
    equal_weight = 1.0 / len(available_tickers)
    
    return {ticker: equal_weight for ticker in available_tickers}

# 2. Define Backtest Parameters
tickers_strategy = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']
start = '2017-01-01'
end = '2024-01-01' # Use current date or recent date
initial_cash = 1000.00
initial_holdings = {}
benchmark = ['SPY']
transaction_cost = 0.001

# 3. Instantiate the Backtester (Float mode active)
backtester = Backtester(
    tickers=tickers_strategy,
    start_date=start,
    end_date=end,
    interval_amount=1,
    interval_unit='D',
    initial_cash=initial_cash,
    initial_holdings=initial_holdings,
    strategy_logic=simple_equal_weight_strategy,
    benchmark_tickers=benchmark,
    transaction_cost_pct=transaction_cost,
    whole_shares_only=False,
    price_type='close'
)

# 4. Run the Simulation
backtester.run()

# 5. Get and Print Results (Operates on float results)
stats = backtester.get_stats(risk_free_rate=0.04)
if stats:
    print("\n--- Performance Results ---")
    print("Strategy Performance:")
    for metric, value in stats.get('strategy', {}).items(): print(f"{metric:>25}: {value:>15.4f}")
    if stats.get('benchmark'):
        print("\nBenchmark Performance:")
        for metric, value in stats.get('benchmark', {}).items(): print(f"{metric:>25}: {value:>15.4f}")
    else: print("\nNo benchmark results calculated.")

equity_df = backtester.get_equity_curve()
if equity_df is not None:
    print("\n--- Equity Curve (Last 5 rows) ---"); print(equity_df.tail())
    backtester.plot_equity_curve(title='Strategy vs Benchmark Performance (Float)', normalize=True)

print("\n--- Backtester Example Finished ---")
```

---

## 📦 requirements.txt

```txt
pandas
numpy
matplotlib
pyarrow
alpaca-py
pytz
```

---

## 📫 Contact

Made by [Saahas Pulivarthi](https://linkedin.com/in/saahas-pulivarthi)  
✉️ saahas.pulivarthi@gmail.com

> This project is part of my ongoing work in quantitative finance. Contributions and feedback welcome!
