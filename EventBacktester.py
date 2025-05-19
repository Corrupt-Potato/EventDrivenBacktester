"""
Single-File Portfolio Backtester using Alpaca Data (Float Version)

Handles data fetching (chunked, sequential), caching, portfolio management,
and backtest simulation using standard floats. Uses end-of-bar timestamps.
"""

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
import pytz
import matplotlib.pyplot as plt
import math
import os
import hashlib # For checksum verification
from enum import Enum
import time # For potential delays if needed

pd.options.mode.chained_assignment = None # 'warn' or None

class AlpacaUnit(Enum):
    """ Maps strings to Alpaca TimeFrameUnit """
    M = TimeFrameUnit.Minute
    H = TimeFrameUnit.Hour
    D = TimeFrameUnit.Day
    MIN = TimeFrameUnit.Minute
    HOUR = TimeFrameUnit.Hour
    DAY = TimeFrameUnit.Day

# --- DataHandler ---
class DataHandler:
    """
    Handles fetching (chunked), preprocessing, caching, and providing market data.
    """
    def __init__(self, tickers, start_date, end_date, interval_amount, interval_unit, cache_dir='data_cache', price_type='close'):
        self.tickers = sorted(list(set(tickers)))
        self.start_date = start_date # Ensure TZ aware
        self.end_date = end_date     # Ensure TZ aware
        self.interval_amount = interval_amount
        self.interval_unit = interval_unit
        self.price_type = price_type
        self.data = None # Holds the final, preprocessed (end-of-bar) data
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        tickers_str = "_".join(self.tickers)
        dates_str = f"{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
        interval_str = f"{self.interval_amount}{self.interval_unit.name}"
        self.cache_filename = f"data_{tickers_str}_{dates_str}_{interval_str}_{self.price_type}.parquet"
        self.cache_filepath = os.path.join(self.cache_dir, self.cache_filename)

        print(f"DataHandler initialized. Tickers: {len(self.tickers)}")
        print(f"Cache file: {self.cache_filepath}")
        print(f"Requested Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Interval: {self.interval_amount} {self.interval_unit.name}, Price Type: {self.price_type}")

    def _fetch_data_chunk(self, client, chunk_start, chunk_end):
        """ Fetches a single chunk of raw data from Alpaca. """
        print(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}...")
        request_params = StockBarsRequest(
            symbol_or_symbols=self.tickers,
            timeframe=TimeFrame(self.interval_amount, self.interval_unit),
            start=chunk_start,
            end=chunk_end,
            adjustment='all',
            feed='sip'
        )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Optional: Add a small delay between requests if hitting rate limits
                # time.sleep(0.2)
                chunk_data = client.get_stock_bars(request_params).df
                return chunk_data
            except Exception as e:
                print(f"  Error fetching chunk {chunk_start.date()}-{chunk_end.date()} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"  Failed to fetch chunk after {max_retries} attempts.")
                    return None # Failed after retries
                time.sleep(1 * (attempt + 1)) # Exponential backoff
        return None

    def _preprocess_raw_data(self, raw_data):
        """ Pivots, fills NA, and shifts timestamps to end-of-bar for raw fetched data. """
        if raw_data is None or raw_data.empty:
            return pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], tz=pytz.UTC, name='timestamp'))

        processed_data = raw_data.reset_index().pivot_table(
            index='timestamp', columns='symbol', values=self.price_type
        )
        # Ensure all requested tickers are columns, even if missing in this chunk/data
        processed_data = processed_data.reindex(columns=self.tickers)

        # --- Preprocessing Steps ---
        print("Applying preprocessing (ffill, bfill, end-of-bar conversion)...")
        processed_data = processed_data.ffill().bfill()

        # Calculate interval_delta for timestamp shift
        unit_map = {TimeFrameUnit.Hour: 'h', TimeFrameUnit.Minute: 'm', TimeFrameUnit.Day: 'D'}
        if self.interval_unit not in unit_map:
            raise ValueError(f"Unsupported TimeFrameUnit for conversion: {self.interval_unit}")
        interval_delta = pd.Timedelta(value=self.interval_amount, unit=unit_map[self.interval_unit])

        # Apply timestamp shift to represent end-of-bar
        if not processed_data.empty:
            processed_data.index = processed_data.index + interval_delta
            print(f"  Timestamp index shifted by {interval_delta} for end-of-bar representation.")
        else:
            print("  Data empty, skipping timestamp shift.")

        return processed_data

    def fetch_data(self):
        """
        Fetches data using Alpaca API (chunked, sequential), preprocesses it,
        and uses cache if available. Stores preprocessed data.
        """
        if os.path.exists(self.cache_filepath):
            print(f"Loading preprocessed data from cache: {self.cache_filepath}")
            try:
                self.data = pd.read_parquet(self.cache_filepath)
                if not isinstance(self.data.index, pd.DatetimeIndex):
                    raise TypeError("Loaded index is not DatetimeIndex!")
                if self.data.index.tz != pytz.UTC: # Ensure timezone consistency
                    print("Warning: Loaded data timezone mismatch. Converting to UTC.")
                    self.data.index = self.data.index.tz_convert(pytz.UTC)
                if set(self.data.columns) != set(self.tickers):
                    print("Warning: Cached columns mismatch. Re-indexing.")
                    self.data = self.data.reindex(columns=self.tickers) # Ensure correct columns

                print(f"  Preprocessed data loaded from cache. Shape: {self.data.shape}")
                self._verify_and_log_checksum(self.data)
                # Data from cache is already preprocessed, no further action needed here
                return
            except Exception as e:
                print(f"Error loading cache file {self.cache_filepath}: {e}. Will fetch fresh data.")
                self.data = None # Reset

        print(f"Fetching '{self.price_type}' data for {len(self.tickers)} tickers (chunked)...")
        combined_raw_data = None
        try:
            # --- Credentials ---
            # Consider environment variables or a config file for security
            ALPACA_API_KEY = "PK3ATC9X95C90176UFYS"
            ALPACA_SECRET_KEY = "HYY5Wu8eGJsVgolPQ9GTLzyUgM1pz0sgjPtQefUH"
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

            # --- Chunking Logic ---
            all_raw_chunks = []
            # Fetch slightly past requested end date to ensure last bar is included before shift
            fetch_end_date = self.end_date + pd.Timedelta(days=1)
            # Generate date ranges (e.g., yearly)
            date_ranges = pd.date_range(
                start=self.start_date.tz_convert(None), # Naive for range generation
                end=fetch_end_date.tz_convert(None),   # Naive for range generation
                freq='12ME', # Adjust chunk size ('6M', '24M', etc.)
                tz=None
            )
            chunk_starts = [self.start_date] + [ts.tz_localize(pytz.UTC) for ts in date_ranges]
            chunk_ends = [ts.tz_localize(pytz.UTC) for ts in date_ranges] + [fetch_end_date]

            request_windows = []
            last_start = None
            for start, end in zip(chunk_starts, chunk_ends):
                # Ensure start < end and avoid redundant requests
                if start < end and (last_start is None or start > last_start):
                    request_windows.append((start, end))
                    last_start = start

            print(f"Total chunks to fetch: {len(request_windows)}")

            for chunk_start, chunk_end in request_windows:
                raw_chunk = self._fetch_data_chunk(client, chunk_start, chunk_end)
                if raw_chunk is not None and not raw_chunk.empty:
                    # Check if required price_type exists before appending
                    if self.price_type in raw_chunk.columns.get_level_values(0):
                        all_raw_chunks.append(raw_chunk)
                    else:
                        print(f"Warning: '{self.price_type}' not found in raw chunk {chunk_start.date()}-{chunk_end.date()}.")

            if not all_raw_chunks:
                print("No data fetched after processing chunks.")
                self.data = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], tz=pytz.UTC, name='timestamp'))
                return

            # Combine raw chunks
            combined_raw_data = pd.concat(all_raw_chunks)
            combined_raw_data = combined_raw_data.sort_index()
            # Remove duplicates potentially arising from chunk boundaries (keep first instance)
            combined_raw_data = combined_raw_data[~combined_raw_data.index.duplicated(keep='first')]
            print(f"Raw data fetched and combined successfully. Shape: {combined_raw_data.shape}")

            # --- Preprocess the combined raw data ---
            self.data = self._preprocess_raw_data(combined_raw_data)

            print(f"Preprocessing complete. Final data shape: {self.data.shape}")
            self._verify_and_log_checksum(self.data) # Checksum preprocessed data

            # --- Save the PREPROCESSED data to Cache ---
            if not self.data.empty:
                try:
                    self.data.to_parquet(self.cache_filepath, index=True)
                    print(f"Preprocessed data saved to cache: {self.cache_filepath}")
                except Exception as e:
                    print(f"Error saving preprocessed data to cache {self.cache_filepath}: {e}")

        except Exception as e:
            print(f"Error during data fetching process: {e}")
            # Ensure self.data is a valid empty DataFrame on error
            self.data = pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], tz=pytz.UTC, name='timestamp'))

    def _verify_and_log_checksum(self, df):
        """ Calculates and prints a checksum for the head of the DataFrame. """
        if df is None or df.empty: return
        try:
            # Use a stable representation for checksumming
            data_head_bytes = df.head().round(6).to_json(orient='split', date_format='iso').encode('utf-8')
            checksum = hashlib.md5(data_head_bytes).hexdigest()
            print(f"  Data head checksum: {checksum}")
        except Exception as chk_e:
            print(f"  Could not calculate checksum: {chk_e}")

    def get_data_for_timestamp(self, timestamp):
        """ Gets preprocessed (end-of-bar) data for a specific timestamp. """
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(pytz.UTC)
        else: timestamp = timestamp.tz_convert(pytz.UTC)

        if self.data is None or self.data.empty:
            # print(f"Warning: Data not available for timestamp {timestamp}") # Can be noisy
            return pd.Series(index=self.tickers, dtype='float64')
        try:
            return self.data.loc[timestamp, self.tickers]
        except KeyError:
             # Fallback: Use asof to get the last available data point AT or BEFORE the timestamp
            try:
                if not self.data.index.is_monotonic_increasing: self.data.sort_index(inplace=True)
                return self.data.asof(timestamp)[self.tickers]
            except Exception as e:
                # print(f"Error using asof for timestamp {timestamp}: {e}") # Can be noisy
                return pd.Series(index=self.tickers, dtype='float64')

    def get_data_up_to_timestamp(self, timestamp):
        """ Gets preprocessed (end-of-bar) data up to and including a specific timestamp. """
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(pytz.UTC)
        else: timestamp = timestamp.tz_convert(pytz.UTC)

        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=self.tickers, index=pd.DatetimeIndex([], tz=pytz.UTC, name='timestamp'))

        if not self.data.index.is_monotonic_increasing: self.data.sort_index(inplace=True)
        # Use .loc slice which includes the end label if it exists
        return self.data.loc[:timestamp, self.tickers]

    def get_full_data(self):
        """ Returns the entire preprocessed (end-of-bar) DataFrame. """
        return self.data

    def get_first_valid_index(self):
        """ Gets the first valid end-of-bar timestamp in the preprocessed data. """
        if self.data is None or self.data.empty:
            return None
        first_valid = self.data.first_valid_index()
        return first_valid.tz_convert(pytz.UTC) if first_valid else None

    def get_last_valid_index(self):
        """ Gets the last valid end-of-bar timestamp in the preprocessed data. """
        if self.data is None or self.data.empty:
            return None
        last_valid = self.data.last_valid_index()
        return last_valid.tz_convert(pytz.UTC) if last_valid else None

# --- Portfolio ---
class Portfolio:
    """ Manages portfolio state (cash, holdings, value) using floats. """
    def __init__(self, initial_cash, initial_holdings={}, start_date=None):
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.holdings = {ticker: float(qty) for ticker, qty in initial_holdings.items()} # {ticker: float_quantity}
        self.positions_value = 0.0
        self.total_value = self.initial_cash
        self.start_date = pd.to_datetime(start_date) # Store for reference
        self.equity_curve = {} # {timestamp: float_total_value}
        self._zero_threshold = 1e-9 # For float comparisons

        print(f"Portfolio initialized: Cash=${self.cash:,.2f}, Holdings={ {k: f'{v:.4f}' for k, v in self.holdings.items()} }")

    def update_time(self, timestamp, current_prices):
        """ Updates portfolio value based on current prices and records equity. """
        pos_value_sum = 0.0
        # Use sorted keys for deterministic calculations
        for ticker in sorted(self.holdings.keys()):
            quantity = self.holdings[ticker]
            price_float = current_prices.get(ticker, np.nan) # Use np.nan for missing prices
            if not pd.isna(price_float) and price_float > 0:
                pos_value_sum += (quantity * price_float)

        self.positions_value = pos_value_sum
        self.total_value = self.cash + self.positions_value
        self.track_performance(timestamp, self.total_value)

    def track_performance(self, timestamp, value):
        """ Records the portfolio's total value at a specific time. """
        self.equity_curve[timestamp] = value

    def get_current_weights(self, current_prices):
        """ Calculates current asset weights based on market value. """
        weights = {}
        current_total_value = self.calculate_value(current_prices)
        if abs(current_total_value) < self._zero_threshold: # Avoid division by zero
            return {ticker: 0.0 for ticker in self.holdings}

        # Calculate weights deterministically
        for ticker in sorted(self.holdings.keys()):
            quantity = self.holdings[ticker]
            price_float = current_prices.get(ticker, np.nan)
            if not pd.isna(price_float) and price_float > 0:
                market_value = quantity * price_float
                weights[ticker] = market_value / current_total_value
            else:
                weights[ticker] = 0.0 # Assign 0 weight if price is invalid

        # Ensure all holdings keys are present, even if weight is 0
        for ticker in self.holdings:
            weights.setdefault(ticker, 0.0)
        return weights

    def calculate_value(self, current_prices):
        """ Calculates total portfolio value (cash + positions). """
        pos_val_sum = 0.0
        for ticker in sorted(self.holdings.keys()):
            quantity = self.holdings[ticker]
            price_float = current_prices.get(ticker, np.nan)
            if not pd.isna(price_float) and price_float > 0:
                pos_val_sum += (quantity * price_float)
        return self.cash + pos_val_sum

    def execute_trade(self, timestamp, ticker, quantity_change, price, transaction_cost_pct):
        """ Executes a buy or sell trade, updating cash and holdings. """
        flt_qty_change = float(quantity_change)
        flt_price = float(price)
        flt_cost_pct = float(transaction_cost_pct)

        if pd.isna(flt_price) or flt_price <= 0:
            print(f"Warning @ {timestamp}: Invalid price ({flt_price}) for {ticker}. Trade aborted.")
            return False

        trade_value = abs(flt_qty_change) * flt_price
        cost = trade_value * flt_cost_pct
        action = 'BUY' if flt_qty_change > 0 else 'SELL'

        # --- Buy Logic ---
        if action == 'BUY':
            total_cost = trade_value + cost
            # Check available cash with tolerance
            if self.cash + self._zero_threshold >= total_cost:
                self.cash -= total_cost
                current_holding = self.holdings.get(ticker, 0.0)
                self.holdings[ticker] = current_holding + flt_qty_change
                # print(f"Executed BUY: {flt_qty_change:.4f} {ticker} @ ${flt_price:.2f}, Cost: ${cost:.2f}")
                return True
            else:
                # print(f"Failed BUY: Insufficient cash for {ticker}. Needed ${total_cost:.2f}, Have ${self.cash:.2f}")
                return False # Insufficient cash

        # --- Sell Logic ---
        elif action == 'SELL':
            current_holding = self.holdings.get(ticker, 0.0)
            # Check available shares with tolerance
            if current_holding + self._zero_threshold >= abs(flt_qty_change):
                # Ensure we don't sell fractional shares beyond what's held due to float issues
                actual_sell_quantity = min(abs(flt_qty_change), current_holding)

                actual_trade_value = actual_sell_quantity * flt_price
                actual_cost = actual_trade_value * flt_cost_pct
                cash_received = actual_trade_value - actual_cost

                self.cash += cash_received
                self.holdings[ticker] = current_holding - actual_sell_quantity

                # Clean up holdings close to zero
                if abs(self.holdings[ticker]) < self._zero_threshold:
                    del self.holdings[ticker]
                # print(f"Executed SELL: {actual_sell_quantity:.4f} {ticker} @ ${flt_price:.2f}, Cost: ${actual_cost:.2f}")
                return True
            else:
                # print(f"Failed SELL: Insufficient shares for {ticker}. Have {current_holding:.4f}, Tried {abs(flt_qty_change):.4f}")
                return False # Insufficient shares
        return False # Should not happen

# --- Backtester ---
class Backtester:
    """ Orchestrates the backtesting process using DataHandler and Portfolio. """
    def __init__(self,
                 tickers, start_date, end_date, interval_amount, interval_unit, initial_cash,
                 strategy_logic, initial_holdings={}, benchmark_tickers=None,
                 benchmark_weights=None, transaction_cost_pct=0.0,
                 whole_shares_only=False, price_type='close'
                ):

        self.strategy_tickers = sorted(list(set(tickers)))
        self.benchmark_tickers = sorted(list(set(benchmark_tickers))) if benchmark_tickers else []
        self.all_tickers = sorted(list(set(self.strategy_tickers + self.benchmark_tickers)))

        self.start_date = pd.to_datetime(start_date).tz_localize(pytz.UTC)
        self.end_date = pd.to_datetime(end_date).tz_localize(pytz.UTC)
        self.interval_amount = int(interval_amount)
        try:
            self.interval_unit_enum = AlpacaUnit[interval_unit.upper()].value # Get the enum value
        except KeyError:
            raise ValueError(f"Invalid interval unit: {interval_unit}. Must be one of {list(AlpacaUnit.__members__)}.")

        self.initial_cash = float(initial_cash)
        self.initial_holdings = {t: float(q) for t, q in initial_holdings.items()}
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.strategy_logic = strategy_logic
        self.benchmark_weights = benchmark_weights
        self.whole_shares_only = whole_shares_only
        self.price_type = price_type

        self._validate_inputs()

        self.data_handler = DataHandler(
            tickers=self.all_tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            interval_amount=self.interval_amount,
            interval_unit=self.interval_unit_enum, # Pass the enum value
            price_type=self.price_type
        )
        self.strategy_portfolio = Portfolio(self.initial_cash, self.initial_holdings, self.start_date)
        self.benchmark_portfolio = None

        self.data = None # Will hold the final sliced data for the simulation period
        self.equity_df = None
        self.stats = None
        self._simulation_run = False
        print("Backtester initialized.")

    def _validate_inputs(self):
        """ Basic validation of input parameters. """
        if not self.strategy_tickers: raise ValueError("Strategy 'tickers' list cannot be empty.")
        if self.start_date >= self.end_date: raise ValueError("Start date must be before end date.")
        if self.initial_cash < 0: raise ValueError("Initial cash cannot be negative.")
        if not callable(self.strategy_logic): raise TypeError("'strategy_logic' must be a function.")
        if self.interval_amount <= 0: raise ValueError("'interval_amount' must be positive.")

        # Validate and normalize benchmark weights if provided
        if self.benchmark_tickers:
            if not self.benchmark_weights: # Assign equal weights if none provided
                print("Benchmark weights not provided, using equal weights.")
                num_bench = len(self.benchmark_tickers)
                self.benchmark_weights = {ticker: 1.0 / num_bench for ticker in self.benchmark_tickers} if num_bench > 0 else {}
            else: # Validate provided weights
                if set(self.benchmark_tickers) != set(self.benchmark_weights.keys()):
                    raise ValueError("Benchmark tickers and benchmark_weights keys do not match.")
                weight_sum = sum(self.benchmark_weights.values())
                if not np.isclose(weight_sum, 1.0):
                    print(f"Warning: Benchmark weights sum ({weight_sum:.4f}) is not 1.0. Normalizing.")
                    self.benchmark_weights = {k: v / weight_sum for k, v in self.benchmark_weights.items()}
        else:
            self.benchmark_weights = {} # Ensure it's an empty dict if no benchmark

    def _initialize_benchmark_portfolio(self, first_timestamp, first_prices):
        """ Initializes the benchmark portfolio based on target weights. """
        if not self.benchmark_tickers: return

        print("Initializing benchmark portfolio...")
        self.benchmark_portfolio = Portfolio(self.initial_cash, start_date=self.start_date)
        initial_total_value = self.initial_cash
        benchmark_holdings = {}

        # Calculate initial holdings based on weights and first prices
        for ticker in self.benchmark_tickers: # Already sorted
            price_float = first_prices.get(ticker, np.nan)
            weight_float = self.benchmark_weights.get(ticker, 0.0)

            if not pd.isna(price_float) and price_float > 0 and weight_float > 0:
                target_value = initial_total_value * weight_float
                quantity = target_value / price_float

                if self.whole_shares_only:
                    quantity = math.floor(quantity)

                if quantity > 0:
                    cost = quantity * price_float # No transaction cost for initial setup
                    if self.benchmark_portfolio.cash >= cost:
                        self.benchmark_portfolio.cash -= cost
                        benchmark_holdings[ticker] = benchmark_holdings.get(ticker, 0.0) + quantity
                    else:
                         print(f"Warning: Insufficient cash for initial benchmark {ticker} ({quantity:.4f} shares).")
            elif weight_float > 0:
                 print(f"Warning: Cannot initialize benchmark for {ticker}. Price missing/invalid ({price_float}).")

        self.benchmark_portfolio.holdings = benchmark_holdings
        # Update value based on actual holdings and track performance at start time
        self.benchmark_portfolio.update_time(first_timestamp, first_prices)
        print(f"Benchmark portfolio initialized. Holdings: { {k: f'{v:.4f}' for k, v in benchmark_holdings.items()} }, Cash: ${self.benchmark_portfolio.cash:,.2f}")

    def _execute_trades(self, timestamp, target_weights, current_prices):
        """ Calculates required trades to match target weights and executes them. """
        if target_weights is None or not isinstance(target_weights, dict):
             # print(f"Debug: No target weights provided at {timestamp}") # Can be noisy
             return # No trades if strategy returns None or invalid weights

        # --- 1. Calculate Target Shares ---
        current_portfolio_value = self.strategy_portfolio.calculate_value(current_prices)
        if current_portfolio_value <= 0: return # Cannot trade if portfolio value is zero or negative

        # Normalize target weights if they don't sum to ~1.0
        total_target_weight = sum(w for w in target_weights.values() if isinstance(w, (int, float)))
        if abs(total_target_weight - 1.0) > 1e-6 and total_target_weight > 1e-9 :
            # print(f"Debug: Normalizing target weights at {timestamp} (Sum={total_target_weight:.4f})") # Optional Debug
            target_weights = {k: v / total_target_weight for k, v in target_weights.items() if isinstance(v, (int, float))}

        target_shares = {} # {ticker: float_quantity}
        # Calculate target shares deterministically
        for ticker in self.strategy_tickers: # Use the defined strategy universe
            weight = target_weights.get(ticker, 0.0) # Default to 0 weight if not specified
            price_float = current_prices.get(ticker, np.nan)

            if weight > 0 and not pd.isna(price_float) and price_float > 0:
                target_value = current_portfolio_value * weight
                raw_target_qty = target_value / price_float
                target_shares[ticker] = math.floor(raw_target_qty) if self.whole_shares_only else raw_target_qty
            elif weight == 0:
                 target_shares[ticker] = 0.0 # Explicitly target zero shares if weight is zero
            # If weight > 0 but price is invalid, we implicitly target current holding by not setting target_shares[ticker] yet. Handled below.

        # --- 2. Determine Required Trades (Difference) ---
        current_holdings = self.strategy_portfolio.holdings.copy()
        all_relevant_tickers = sorted(list(set(current_holdings.keys()) | set(target_shares.keys()) | set(target_weights.keys()) ))
        trades_to_make = {} # {ticker: float_quantity_change}

        for ticker in all_relevant_tickers:
            current_qty = current_holdings.get(ticker, 0.0)
            # If target_shares wasn't set (e.g., invalid price), assume target is current holding (no trade)
            target_qty = target_shares.get(ticker, current_qty if ticker in current_holdings else 0.0)

            share_diff = target_qty - current_qty

            # Only consider trade if difference is significant or whole shares required
            if abs(share_diff) > self.strategy_portfolio._zero_threshold:
                price_float = current_prices.get(ticker, np.nan)
                # Check if price is valid before adding to trades
                if not pd.isna(price_float) and price_float > 0:
                    # Optional: Add minimum trade value threshold?
                    trades_to_make[ticker] = share_diff
                # else: print(f"Debug: Skipping trade for {ticker} at {timestamp} due to invalid price.") # Optional

        # --- 3. Execute Sells First ---
        sell_tickers = sorted([ticker for ticker, diff in trades_to_make.items() if diff < 0])
        for ticker in sell_tickers:
            share_diff = trades_to_make[ticker]
            price_float = current_prices.get(ticker) # Already checked for validity
            self.strategy_portfolio.execute_trade(timestamp, ticker, share_diff, price_float, self.transaction_cost_pct)

        # --- 4. Calculate Buys and Apply Cash Constraint ---
        buy_orders_potential = []
        potential_buy_tickers = sorted([ticker for ticker, diff in trades_to_make.items() if diff > 0])
        for ticker in potential_buy_tickers:
            diff = trades_to_make[ticker]
            price_float = current_prices.get(ticker) # Already checked for validity
            buy_orders_potential.append({'ticker': ticker, 'qty': diff, 'price': price_float})

        if not buy_orders_potential: return # No buys needed

        # Calculate total cash needed for all potential buys
        total_cash_needed_for_buys = 0.0
        for buy in buy_orders_potential:
             trade_value = buy['qty'] * buy['price']
             cost = trade_value * self.transaction_cost_pct
             total_cash_needed_for_buys += (trade_value + cost)

        cash_available = self.strategy_portfolio.cash
        reduction_factor = 1.0

        # Calculate reduction factor if short on cash (with tolerance)
        epsilon = 1e-9 # Tolerance for float comparison
        if total_cash_needed_for_buys > cash_available + epsilon:
             if cash_available > epsilon: # Avoid division by zero or near-zero
                  reduction_factor = cash_available / total_cash_needed_for_buys
             else:
                  reduction_factor = 0.0 # Cannot buy anything
             # print(f"Debug: Reducing buy orders at {timestamp} by factor {reduction_factor:.4f}") # Optional Debug

        # Apply reduction factor (capped at 1.0)
        actual_buy_orders = []
        min_qty_threshold = 1e-9 if not self.whole_shares_only else 0.0 # Allow 0 whole shares

        for buy in buy_orders_potential:
            adjusted_qty = buy['qty'] * reduction_factor
            if self.whole_shares_only:
                 # Floor the adjusted quantity if reduced, otherwise keep original integer target if factor is 1.0
                 adjusted_qty = math.floor(adjusted_qty) if reduction_factor < 1.0 else math.floor(buy['qty'])

            if adjusted_qty > min_qty_threshold: # Only add if adjusted quantity is meaningful
                 actual_buy_orders.append({
                     'ticker': buy['ticker'], 'qty': adjusted_qty, 'price': buy['price']
                 })

        # --- 5. Execute Actual (Potentially Reduced) Buys ---
        for buy in actual_buy_orders: # Already sorted
            self.strategy_portfolio.execute_trade(timestamp, buy['ticker'], buy['qty'], buy['price'], self.transaction_cost_pct)

    def _prepare_results(self):
        """ Consolidates equity curves into a DataFrame. """
        print("Preparing results...")
        strategy_equity = self.strategy_portfolio.equity_curve
        benchmark_equity = self.benchmark_portfolio.equity_curve if self.benchmark_portfolio else {}

        equity_dict = {'Strategy': strategy_equity}
        if benchmark_equity:
            equity_dict['Benchmark'] = benchmark_equity

        # Create DataFrame directly from dict of timestamps->values
        self.equity_df = pd.DataFrame(equity_dict)
        self.equity_df.index = pd.to_datetime(self.equity_df.index) # Ensure DatetimeIndex
        self.equity_df.sort_index(inplace=True) # Sort by timestamp
        # Forward fill to handle potential gaps if benchmark/strategy didn't update simultaneously
        self.equity_df = self.equity_df.ffill()
        print("Results prepared.")

    def _calculate_stats(self, equity_series, risk_free_rate=0.0):
        """ Calculates performance statistics for a given equity curve Series. """
        stats = { # Default values
            'Total Return (%)': 0.0, 'CAGR (%)': 0.0, 'Annualized Volatility': 0.0,
            'Annualized Sharpe Ratio': np.nan, 'Annualized Sortino Ratio': np.nan,
            'Max Drawdown (%)': 0.0, 'Annualized Calmar Ratio': np.nan, 'Final Value': 0.0
        }
        if equity_series.empty or equity_series.isnull().all(): return stats
        # Drop NaNs and ensure enough data points
        equity_series = equity_series.dropna()
        if len(equity_series) < 2: return stats

        final_value = equity_series.iloc[-1]
        initial_value = equity_series.iloc[0]
        stats['Final Value'] = final_value

        if initial_value <= 1e-9: # Avoid division by zero or near-zero
             print("Warning: Initial equity value is near zero. Stats may be unreliable.")
             stats['Total Return (%)'] = np.inf if final_value > initial_value else -100.0
             return stats

        # Calculate returns
        total_return = (final_value / initial_value) - 1.0
        returns = equity_series.pct_change().dropna()
        if returns.empty: # Handle case where returns can't be calculated (e.g., constant equity)
             stats['Total Return (%)'] = total_return * 100
             return stats

        # Calculate time period in years
        years = max((equity_series.index[-1] - equity_series.index[0]).days / 365.25, 1e-12) # Avoid division by zero days

        # CAGR
        cagr = ((final_value / initial_value) ** (1.0 / years)) - 1.0

        # Assuming daily-like frequency for annualization (adjust if needed)
        # Heuristic: Count unique days and estimate points per year
        days_in_data = (equity_series.index[-1] - equity_series.index[0]).days
        points_per_year = len(returns) / years if years > 0 else 252 # Default to 252 if years is ~0
        annualization_factor = np.sqrt(max(points_per_year, 1)) # Avoid sqrt(0)

        # Volatility
        annualized_volatility = returns.std() * annualization_factor

        # Sharpe Ratio
        risk_free_rate_periodic = (1 + float(risk_free_rate))**(1/points_per_year) - 1
        excess_returns_mean = returns.mean() - risk_free_rate_periodic
        excess_returns_mean_annual = excess_returns_mean * points_per_year
        if abs(annualized_volatility) > 1e-9:
             sharpe_ratio = excess_returns_mean_annual / annualized_volatility
        else: sharpe_ratio = np.nan

        # Sortino Ratio
        downside_returns = returns[returns < risk_free_rate_periodic] # Or returns < 0 ? Using RFR here.
        if not downside_returns.empty:
            downside_deviation = downside_returns.std() * annualization_factor
            if abs(downside_deviation) > 1e-9:
                sortino_ratio = excess_returns_mean_annual / downside_deviation
            else: sortino_ratio = np.nan # Or inf if mean excess return > 0? Setting nan.
        else: sortino_ratio = np.nan # No downside returns observed

        # Max Drawdown
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() # Already a percentage (or fraction)

        # Calmar Ratio
        if abs(max_drawdown) > 1e-9:
            calmar_ratio = cagr / abs(max_drawdown)
        else: calmar_ratio = np.nan

        stats.update({
            'Total Return (%)': total_return * 100,
            'CAGR (%)': cagr * 100,
            'Annualized Volatility': annualized_volatility,
            'Annualized Sharpe Ratio': sharpe_ratio,
            'Annualized Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Annualized Calmar Ratio': calmar_ratio
        })
        return stats

    # --- Public Methods ---
    def run(self):
        """ Public method to start the backtest simulation. """
        if self._simulation_run:
            print("Simulation has already been run. Re-instantiate Backtester to run again.")
            return
        self._run_simulation()

    def _run_simulation(self):
        """ Executes the core backtesting loop using end-of-bar timestamps. """
        print("\n--- Starting Backtest Simulation ---")
        self.data_handler.fetch_data() # Fetches or loads preprocessed data
        full_data = self.data_handler.get_full_data() # Gets the end-of-bar, preprocessed data

        if full_data is None or full_data.empty:
            print("Error: No data available after fetching/loading.")
            self._simulation_run = False
            return

        # --- Determine simulation period based on available data and request ---
        first_available_ts = self.data_handler.get_first_valid_index()
        last_available_ts = self.data_handler.get_last_valid_index()

        if first_available_ts is None or last_available_ts is None:
            print("Error: No valid data points found in the data.")
            self._simulation_run = False
            return

        # Calculate the *start* time of the first available bar
        unit_map = {TimeFrameUnit.Hour: 'h', TimeFrameUnit.Minute: 'm', TimeFrameUnit.Day: 'D'}
        interval_delta = pd.Timedelta(value=self.interval_amount, unit=unit_map[self.interval_unit_enum])
        first_bar_start_time = first_available_ts - interval_delta

        # Effective start is the later of requested start or first bar's start
        sim_start_date = max(self.start_date, first_bar_start_time)
        # Effective end is the earlier of requested end or last available bar's end
        sim_end_date = min(self.end_date, last_available_ts)

        if first_bar_start_time > self.start_date:
            print(f"Adjusting simulation start: Using first available data from {first_bar_start_time.date()}")
        if last_available_ts < self.end_date:
            print(f"Adjusting simulation end: Using last available data up to {last_available_ts.date()}")

        # --- Slice the preprocessed data for the exact simulation period ---
        self.data = full_data.loc[sim_start_date : sim_end_date] # Use .loc for label slicing

        if self.data.empty:
            print(f"Error: No data remains after slicing for period: {sim_start_date} to {sim_end_date}")
            self._simulation_run = False
            return

        print(f"Simulation period (End-of-Bar Timestamps): {self.data.index[0]} to {self.data.index[-1]}")

        # --- Initialization ---
        first_timestamp = self.data.index[0]
        first_prices = self.data.loc[first_timestamp]

        # Update portfolio value at the very start & init benchmark
        self.strategy_portfolio.update_time(first_timestamp, first_prices)
        if self.benchmark_tickers:
            self._initialize_benchmark_portfolio(first_timestamp, first_prices)

        # --- Simulation Loop ---
        print(f"Running simulation loop for {len(self.data.index)-1} steps...")
        # Iterate from the second timestamp onwards for trading logic
        for i in range(1, len(self.data.index)):
            timestamp = self.data.index[i]       # Current end-of-bar timestamp
            current_prices = self.data.iloc[i]    # Prices known at the end of this bar
            # previous_timestamp = self.data.index[i-1] # Timestamp of the previous bar end

            # Check for NaN prices in relevant tickers *before* doing anything else
            assets_needed = list(set(list(self.strategy_portfolio.holdings.keys()) + self.strategy_tickers + self.benchmark_tickers))
            valid_assets_needed = [tk for tk in assets_needed if tk in current_prices.index]
            if current_prices[valid_assets_needed].isnull().any():
                print(f"Warning @ {timestamp}: NaN price detected for required assets. Skipping trades, updating value only.")
                # Update value based on available prices, even if some are NaN
                self.strategy_portfolio.update_time(timestamp, current_prices)
                if self.benchmark_portfolio: self.benchmark_portfolio.update_time(timestamp, current_prices)
                continue # Skip strategy logic and trading for this step

            # 1. Update portfolio value with current prices *before* trading
            self.strategy_portfolio.update_time(timestamp, current_prices)
            if self.benchmark_portfolio: self.benchmark_portfolio.update_time(timestamp, current_prices)

            # 2. Get data *up to this timestamp* for the strategy logic
            # This includes the prices/data known at the end of the current bar
            data_for_strategy = self.data_handler.get_data_up_to_timestamp(timestamp)

            # 3. Call strategy logic function
            # Strategy sees data finalized at 'timestamp'
            target_weights = self.strategy_logic(timestamp, data_for_strategy, self.strategy_portfolio)

            # 4. Execute trades based on strategy output and prices finalized at 'timestamp'
            # Trades occur notionally *after* the bar closes / at the start of the next bar
            if target_weights is not None:
                self._execute_trades(timestamp, target_weights, current_prices)

        print("Simulation loop complete.")
        self._prepare_results()
        self._simulation_run = True
        print("--- Backtest Simulation Finished ---")

    def get_stats(self, risk_free_rate=0.0):
        """ Calculates and returns performance statistics. """
        if not self._simulation_run: print("Error: Simulation not run yet."); return None
        if self.equity_df is None or self.equity_df.empty: print("Error: Equity curve data not available."); return None

        print(f"\n--- Calculating Performance Statistics (RFR={risk_free_rate:.2%}) ---")
        self.stats = {}
        self.stats['strategy'] = self._calculate_stats(self.equity_df['Strategy'], risk_free_rate)
        if 'Benchmark' in self.equity_df.columns:
            self.stats['benchmark'] = self._calculate_stats(self.equity_df['Benchmark'], risk_free_rate)
        else: self.stats['benchmark'] = {}

        return self.stats

    def get_equity_curve(self):
        """ Returns the portfolio equity curve DataFrame. """
        if not self._simulation_run: print("Error: Simulation not run yet."); return None
        return self.equity_df

    def plot_equity_curve(self, title='Portfolio Performance', normalize=True):
        """ Plots the strategy and benchmark equity curves. """
        if self.equity_df is None or self.equity_df.empty: print("Error: No equity curve data to plot."); return

        print("\n--- Plotting Equity Curve ---")
        plot_df = self.equity_df.copy()
        # Ensure data is suitable for plotting (e.g., fill small gaps)
        plot_df = plot_df.ffill().dropna(how='all')
        if plot_df.empty: print("Error: Equity data is empty or all NaN after fill."); return

        # Normalize if requested
        if normalize:
            first_valid_idx = plot_df.first_valid_index()
            if first_valid_idx is not None and abs(plot_df.loc[first_valid_idx].min()) > 1e-9 : # Check valid index and non-zero start
                 plot_df = (plot_df / plot_df.loc[first_valid_idx]) * 100
            else:
                 print("Warning: Could not normalize equity curve (invalid start point or near-zero value). Plotting raw values.")
                 normalize = False # Fallback to non-normalized plot

        # Plotting
        plt.style.use('seaborn-v0_8-darkgrid') # Or choose another style
        fig, ax = plt.subplots(figsize=(12, 7))

        if 'Strategy' in plot_df.columns:
            ax.plot(plot_df.index, plot_df['Strategy'], label='Strategy', linewidth=2, color='blue')
        if 'Benchmark' in plot_df.columns:
            ax.plot(plot_df.index, plot_df['Benchmark'], label='Benchmark', linestyle='--', linewidth=1.5, color='grey')

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value' + (' (Normalized Start=100)' if normalize else ''), fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
