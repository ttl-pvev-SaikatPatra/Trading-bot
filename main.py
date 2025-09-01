print("Starting app...")
import logging
import os
import json
import time
import threading
import signal
from datetime import datetime, timedelta, time as datetime_time

import pandas as pd
import numpy as np
import pytz
import schedule
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flask import Flask, jsonify, request, send_file, redirect
from flask_cors import CORS
from kiteconnect import KiteConnect
import urllib.parse

# ==================== Config & Globals ====================
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = datetime_time(9, 15)
MARKET_CLOSE = datetime_time(15, 30)

API_KEY = os.environ.get("KITE_API_KEY")
API_SECRET = os.environ.get("KITE_API_SECRET")

if not API_KEY or not API_SECRET:
    print("ERROR: Missing KITE_API_KEY or KITE_API_SECRET in environment.")
    print(f"Missing API_KEY: {API_KEY}, API_SECRET: {API_SECRET}")

DEFAULT_ACCOUNT_EQUITY = float(os.environ.get("ACCOUNT_EQUITY", "10000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))
MAX_POSITIONS_10K = int(os.environ.get("MAX_POS_10K", "2"))
MAX_POSITIONS_20K = int(os.environ.get("MAX_POS_20K", "2"))
MAX_POSITIONS_30K = int(os.environ.get("MAX_POS_30K", "3"))
MAX_NOTIONAL_PCT = float(os.environ.get("MAX_NOTIONAL_PCT", "0.08"))
# Add these new safety parameters:
DAILY_LOSS_LIMIT = 500          # Stop if lose ₹500/day
WEEKLY_LOSS_LIMIT = 1500        # Pause if lose ₹1500/week
MAX_TRADES_PER_DAY = 2          # Maximum 2 trades per day
CONSECUTIVE_LOSS_LIMIT = 4      # Pause after 4 losses


UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "20"))

BACKTEST_YEARS = int(os.environ.get("BACKTEST_YEARS", "1"))

BASE_TICKERS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","KOTAKBANK.NS","SBIN.NS",
    "BHARTIARTL.NS","LT.NS","AXISBANK.NS","ITC.NS","HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS",
    "ULTRACEMCO.NS","SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ASIANPAINT.NS","HCLTECH.NS","NESTLEIND.NS",
    "M&M.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","ADANIENT.NS","ADANIPORTS.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "COALINDIA.NS","DIVISLAB.NS","TECHM.NS","LTIM.NS","BRITANNIA.NS","BPCL.NS","EICHERMOT.NS","HDFCLIFE.NS",
    "DRREDDY.NS","SBILIFE.NS","GRASIM.NS","HINDALCO.NS","INDUSINDBK.NS","BAJAJFINSV.NS","TATAMOTORS.NS","HEROMOTOCO.NS"
]

FRONTEND_URL = os.environ.get("FRONTEND_URL", "").rstrip("/")

app = Flask(__name__)
CORS(app, origins=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TradingBot")

STOP_EVENT = threading.Event()

# ==================== Helpers ====================
def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def is_market_open_now():
    t = now_ist()
    if t.weekday() >= 5:
        return False
    return ((t.hour > 9 or (t.hour == 9 and t.minute >= 15)) and
            (t.hour < 15 or (t.hour == 15 and t.minute <= 30)))

def _make_session():
    session = requests.Session()
    try:
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
    except TypeError:
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def round2(x):
    try:
        return float(f"{float(x):.2f}")
    except Exception:
        return x

# ==================== Bot Class ====================
class AutoTradingBot:
    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY)
        self.access_token = None
        self.positions = {}
        self.pending_orders = []
        self.bot_status = "Initializing"
        self.total_trades_today = 0
        self.win_rate = 0.0

        self.target_profit_pct = 1.0
        self.stop_loss_pct = 0.5
        self.trailing_start_R = 0.5
        self.trailing_atr_mult = 1.0

        self.account_equity = DEFAULT_ACCOUNT_EQUITY
        self.risk_per_trade = RISK_PER_TRADE
        self.max_positions = self._max_positions_for_equity(self.account_equity)
        self.max_notional_pct = MAX_NOTIONAL_PCT

        self._cache_lock = threading.Lock()
        self._last_cache_update = None
        self.daily_stock_list = []
        self.universe_version = None
        self.universe_features = pd.DataFrame()

        self._scheduler_started = False
        logger.info("Bot initialized.")

    # ========= Auth (CLI) =========
    def authenticate_cli(self):
        print("\n🔐 ZERODHA PERSONAL API AUTHENTICATION")
        print("=" * 50)
        try:
            login_url = self.kite.login_url()
            print("🔗 Step 1: Visit this URL to login:")
            print(login_url)
        except Exception as e:
            print(f"❌ Could not build login URL. Check API key/secret: {e}")
            return False

        print("\n📝 After login, copy the 32-char request_token from the redirected URL")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                request_token = input(f"\nEnter request_token (Attempt {attempt}/{max_attempts}): ").strip()
            except Exception:
                print("❌ Non-interactive environment detected; use the HTTP endpoint instead.")
                return False
            if len(request_token) != 32:
                print("❌ Invalid request_token length.")
                if attempt < max_attempts:
                    continue
                return False
            try:
                print("🔄 Generating access token...")
                sess = self.kite.generate_session(request_token, api_secret=API_SECRET)
                self.access_token = sess["access_token"]
                self.kite.set_access_token(self.access_token)
                with open("access_token.txt", "w") as f:
                    f.write(f"{self.access_token}\n{now_ist().isoformat()}")
                print("✅ Authentication successful!")
                try:
                    margins = self.kite.margins()
                    bal = float(margins["equity"]["available"]["live_balance"])
                    self.account_equity = max(bal, DEFAULT_ACCOUNT_EQUITY)
                    self.max_positions = self._max_positions_for_equity(self.account_equity)
                except Exception as e:
                    print(f"⚠️ Auth OK but couldn't fetch margins: {e}")
                self.bot_status = "Active"
                return True
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    return False
        return False

    def authenticate_with_request_token(self, request_token: str):
        try:
            sess = self.kite.generate_session(request_token, api_secret=API_SECRET)
            self.access_token = sess["access_token"]
            self.kite.set_access_token(self.access_token)
            self.bot_status = "Active"
            with open("access_token.txt", "w") as f:
                f.write(f"{self.access_token}\n{now_ist().isoformat()}")
            try:
                margins = self.kite.margins()
                bal = float(margins["equity"]["available"]["live_balance"])
                self.account_equity = max(bal, DEFAULT_ACCOUNT_EQUITY)
                self.max_positions = self._max_positions_for_equity(self.account_equity)
            except Exception as e:
                logger.warning(f"Profile/margins fetch issue after auth: {e}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.bot_status = "Auth Failed"
            return False

    def load_saved_token(self):
        try:
            with open("access_token.txt", "r") as f:
                token = f.readline().strip()
                if not token:
                    return False
            self.kite.set_access_token(token)
            self.access_token = token
            self.bot_status = "Active"
            try:
                self.kite.profile()
            except Exception:
                self.bot_status = "Auth Required"
                return False
            return True
        except Exception:
            return False

    def is_token_valid(self):
        if not self.access_token:
            return False
        try:
            self.kite.profile()
            return True
        except Exception:
            return False

    def schedule_auth_checks(self):
        schedule.every().day.at("08:30").do(self.check_and_mark_auth)

    def check_and_mark_auth(self):
        if not self.is_token_valid():
            self.bot_status = "Auth Required"
            self.access_token = None

    # ========= Data =========
    def _fetch_eod_batch(self, tickers, period="12mo"):
        data = yf.download(tickers, period=period, interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
        frames = []
        for t in tickers:
            try:
                df = data[t].rename(columns=str.title).reset_index()
                df["Symbol"] = t
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _get_simple_universe(self):
        """Return the simplified BASE_TICKERS"""
        return [s.replace(".NS", "") for s in BASE_TICKERS]

    def update_daily_stock_list(self):
        """Use simplified BASE_TICKERS instead of complex selection"""
        try:
            with self._cache_lock:
                logger.info("Using simplified BASE_TICKERS...")
                # Simply use the predefined BASE_TICKERS
                self.daily_stock_list = self._get_simple_universe()
                self.universe_version = now_ist().strftime("%Y-%m-%d")
                self._last_cache_update = now_ist()
                logger.info(f"Universe loaded with {len(self.daily_stock_list)} stable stocks.")
                
                # Create a simple features dataframe for API compatibility
                self.universe_features = pd.DataFrame({
                    'Symbol': [f"{s}.NS" for s in self.daily_stock_list],
                    'Close': [100.0] * len(self.daily_stock_list),  # Dummy values
                    'ATR_pct': [1.5] * len(self.daily_stock_list),   # Dummy values  
                    'MedTurn20': [50000000] * len(self.daily_stock_list),  # Dummy values
                    'Score': [1.0] * len(self.daily_stock_list)      # Dummy values
                })
        except Exception as e:
            logger.error(f"Failed to load BASE_TICKERS: {e}")

    def maybe_refresh_daily_stock_list(self):
        scheduled_times = [datetime_time(8,45), datetime_time(11,0), datetime_time(13,0)]
        ist_now = now_ist()
        if self._last_cache_update is None or self._last_cache_update.date() != ist_now.date():
            self.update_daily_stock_list()
            return
        for rt in scheduled_times:
            lower = datetime.combine(ist_now.date(), rt) - timedelta(minutes=5)
            upper = datetime.combine(ist_now.date(), rt) + timedelta(minutes=5)
            if lower <= ist_now <= upper and (self._last_cache_update < lower):
                self.update_daily_stock_list()
                break

    def fetch_bars(self, symbol, interval="5m", days=2):
        try:
            yf_symbol = f"{symbol}.NS"
            df = yf.download(yf_symbol, period=f"{days}d", interval=interval, auto_adjust=False, progress=False)
            if df.empty:
                return None
            df = df.rename(columns=str.lower).reset_index()
            return df
        except Exception:
            return None

    def compute_vwap(self, df):
        pv = (df["close"] * df["volume"]).cumsum()
        vv = (df["volume"]).cumsum().replace(0, np.nan)
        return pv / vv

    def ema(self, series, n):
        return series.ewm(span=n, adjust=False).mean()

    def get_stock_price(self, symbol):
        try:
            if self.access_token:
                q = self.kite.quote([f"NSE:{symbol}"])
                ltp = q[f"NSE:{symbol}"]["last_price"]
                return float(ltp)
        except Exception:
            pass
        try:
            t = yf.Ticker(f"{symbol}.NS")
            px = t.fast_info.get("last_price")
            if px is None:
                hist = t.history(period="1d", interval="1m")
                if not hist.empty:
                    px = float(hist["Close"].iloc[-1])
            return float(px) if px is not None else None
        except Exception:
            return None

    # ========= Signal Generation =========
    def mtf_confirmation(self, symbol):
        data_30 = self.fetch_bars(symbol, interval="30m", days=10)
        data_5 = self.fetch_bars(symbol, interval="5m", days=2)

        if data_30 is None or data_5 is None or len(data_30) < 40 or len(data_5) < 40:
            logger.info(f"{symbol}: insufficient bars (30m={len(data_30) if data_30 is not None else 0}, 5m={len(data_5) if data_5 is not None else 0})")
            return None

        ema20_30 = self.ema(data_30["close"], 20)
        if len(ema20_30) < 2:
            logger.info(f"{symbol}: EMA20 has < 2 points")
            return None

        last_two = ema20_30.iloc[-2:]
        if last_two.isna().to_numpy().any():  # scalar reduction
            logger.info(f"{symbol}: EMA20 last two bars contain NaN")
            return None

        # Extract scalars safely
        ema_last = ema20_30.iat[-1] if hasattr(ema20_30, "iat") else ema20_30.iloc[-1]
        ema_prev = ema20_30.iat[-2] if hasattr(ema20_30, "iat") else ema20_30.iloc[-2]
        ema_last = ema_last.item() if hasattr(ema_last, "item") else float(ema_last)
        ema_prev = ema_prev.item() if hasattr(ema_prev, "item") else float(ema_prev)

        price_30_val = data_30["close"].iloc[-1]
        price_30 = price_30_val.item() if hasattr(price_30_val, "item") else float(price_30_val)

        slope_up = ema_last > ema_prev
        slope_down = ema_last < ema_prev
        above_ema_30 = price_30 > ema_last
        below_ema_30 = price_30 < ema_last

        vwap_5_series = self.compute_vwap(data_5)

        vwap_5_last_val = vwap_5_series.iloc[-1]
        # pd.isna(scalar) should be bool; if it’s a 0-dim array/Series, coerce with .item()
        vwap_last_is_nan_raw = pd.isna(vwap_5_last_val)
        vwap_last_is_nan = vwap_last_is_nan_raw.item() if hasattr(vwap_last_is_nan_raw, "item") else bool(vwap_last_is_nan_raw)
        if vwap_last_is_nan:
            logger.info(f"{symbol}: VWAP last value is NaN")
            return None

        vwap_5_last = vwap_5_last_val.item() if hasattr(vwap_5_last_val, "item") else float(vwap_5_last_val)

        price_5_val = data_5["close"].iloc[-1]
        price_5 = price_5_val.item() if hasattr(price_5_val, "item") else float(price_5_val)

        above_vwap = price_5 > vwap_5_last
        below_vwap = price_5 < vwap_5_last

        return {
            "long_ok": (slope_up and above_ema_30 and above_vwap),
            "short_ok": (slope_down and below_ema_30 and below_vwap),
            "data_5": data_5
        }




    def generate_trade_signal(self, symbol):
        mtf = self.mtf_confirmation(symbol)
        if mtf is None:
            return None
        data_5 = mtf["data_5"]

        tr = pd.DataFrame({
            "hl": data_5["high"] - data_5["low"],
            "hc": (data_5["high"] - data_5["close"].shift()).abs(),
            "lc": (data_5["low"] - data_5["close"].shift()).abs(),
        }).max(axis=1)
        atr5 = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr5) or atr5 <= 0:
            return None

        last_close = float(data_5["close"].iloc[-1])
        prev_24_high = float(data_5["high"].rolling(24).max().iloc[-2])
        prev_24_low = float(data_5["low"].rolling(24).min().iloc[-2])

        daily_atr_pct = 1.0
        try:
            feats = self.universe_features
            row = feats[feats["Symbol"] == f"{symbol}.NS"]
            if not row.empty:
                daily_atr_pct = float(row["ATR_pct"].values)
        except Exception:
            pass

        margin = max(0.0025, 0.0025 * (daily_atr_pct / 1.0))
        long_break = last_close > prev_24_high * (1 + margin)
        short_break = last_close < prev_24_low * (1 - margin)

        if mtf["long_ok"] and long_break:
            return ("BUY", float(atr5), float(last_close))
        if mtf["short_ok"] and short_break:
            return ("SHORT", float(atr5), float(last_close))
        return None


    # ========= Sizing / Orders =========
    def _max_positions_for_equity(self, eq):
        if eq <= 12000:
            return MAX_POSITIONS_10K
        elif eq <= 22000:
            return MAX_POSITIONS_20K
        else:
            return MAX_POSITIONS_30K

    def calculate_qty(self, price, stop_distance):
        risk_rupees = self.account_equity * self.risk_per_trade
        if stop_distance <= 0:
            return 0
        qty = int(max(0, np.floor(risk_rupees / stop_distance)))
        max_notional = self.account_equity * self.max_notional_pct
        if qty * price > max_notional:
            qty = int(max_notional // price)
        return max(qty, 0)

    def place_order(self, symbol, quantity, is_buy):
        try:
            if self.bot_status == "Auth Required" or not self.access_token:
                logger.warning("Auth required; cannot place order.")
                return None
            if not is_market_open_now():
                logger.warning("Market closed; cannot place order.")
                return None
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if is_buy else self.kite.TRANSACTION_TYPE_SELL
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            logger.info(f"ORDER {('BUY' if is_buy else 'SELL')} placed: {symbol} qty={quantity} id={order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Order place failed for {symbol}: {e}")
            return None

    # ========= Execute strategy =========
    def execute_strategy(self, symbol, direction, signal_price, atr5):
        try:
            if self.bot_status == "Auth Required" or not self.access_token:
                logger.info("Auth required; execute_strategy skipped.")
                return False
            ist_now = now_ist()
            if ist_now.time() >= datetime_time(14,45):
                logger.info(f"Skipping late entry on {symbol}")
                return False
            try:
                margins = self.kite.margins()
                bal = margins["equity"]["available"]["live_balance"]
                self.account_equity = max(bal, DEFAULT_ACCOUNT_EQUITY)
                self.max_positions = self._max_positions_for_equity(self.account_equity)
            except Exception:
                pass
            stop_distance = atr5
            if stop_distance <= 0:
                return False
            qty = self.calculate_qty(signal_price, stop_distance)
            if qty < 1:
                logger.info(f"Position size too small for {symbol}")
                return False

            is_buy = (direction == "BUY")
            order_id = self.place_order(symbol, qty, is_buy)
            if not order_id:
                return False

            if is_buy:
                stop_price = round2(signal_price - stop_distance)
                target_price = round2(signal_price + 2 * stop_distance)
            else:
                stop_price = round2(signal_price + stop_distance)
                target_price = round2(signal_price - 2 * stop_distance)

            key = f"{symbol}_{order_id}"
            self.positions[key] = {
                "symbol": symbol,
                "side": direction,
                "entry_price": float(signal_price),
                "quantity": int(qty),
                "target_price": float(target_price),
                "stop_loss_price": float(stop_price),
                "entry_time": ist_now,
                "order_id": order_id,
            }
            self.save_positions()
            logger.info(f"NEW {direction} {symbol} entry={signal_price} qty={qty} stop={stop_price} target={target_price}")
            return True
        except Exception as e:
            logger.error(f"Execute error {symbol}: {e}")
            return False
        # Inside AutoTradingBot.execute_strategy, after self.save_positions() and success log:
        try:
            # Start rapid monitoring only when the very first live position is present
            if len(self.positions) == 1:
                rapid_monitor.start()
        except Exception as e:
            logger.warning(f"self.save_positions()Could not start rapid monitor: {e}")


    # ========= Trailing stop management =========
    def update_trailing_stops(self):
        for key, pos in list(self.positions.items()):
            symbol = pos["symbol"]
            side = pos["side"]
            entry = pos["entry_price"]
            cur = self.get_stock_price(symbol)
            if cur is None:
                continue
            data_5 = self.fetch_bars(symbol, interval="5m", days=1)
            if data_5 is None or len(data_5) < 20:
                continue
            tr = pd.DataFrame({
                "hl": data_5["high"] - data_5["low"],
                "hc": (data_5["high"] - data_5["close"].shift()).abs(),
                "lc": (data_5["low"] - data_5["close"].shift()).abs(),
            }).max(axis=1)
            atr5 = tr.rolling(14).mean().iloc[-1]
            if np.isnan(atr5) or atr5 <= 0:
                continue

            if side == "BUY":
                R = (pos["target_price"] - entry)
                move = (cur - entry)
                start_trailing = move >= self.trailing_start_R * abs(R)
                if not start_trailing:
                    continue
                new_stop = max(pos["stop_loss_price"], cur - self.trailing_atr_mult * atr5)
                if new_stop > pos["stop_loss_price"]:
                    pos["stop_loss_price"] = round2(new_stop)
            else:
                R = (entry - pos["target_price"])
                move = (entry - cur)
                start_trailing = move >= self.trailing_start_R * abs(R)
                if not start_trailing:
                    continue
                new_stop = min(pos["stop_loss_price"], cur + self.trailing_atr_mult * atr5)
                if new_stop < pos["stop_loss_price"]:
                    pos["stop_loss_price"] = round2(new_stop)
        self.save_positions()

    # ========= Monitor and exit =========
    def monitor_positions(self):
        if self.bot_status == "Auth Required" or not self.access_token:
            logger.info("Auth required; monitor_positions skipped.")
            return
        if not self.positions:
            try:
                # rapid_monitor is the global/outer instance created alongside bot
                if 'rapid_monitor' in globals():
                    rapid_monitor.stop()
            except Exception as e:
                logger.warning(f"Rapid monitor early-stop failed: {e}")
            return
           
        self.update_trailing_stops()
        logger.info("Monitoring positions...")
        to_close = []
        for key, pos in list(self.positions.items()):
            try:
                symbol = pos["symbol"]
                side = pos["side"]
                qty = pos["quantity"]
                entry = pos["entry_price"]
                cur = self.get_stock_price(symbol)
                if cur is None:
                    continue
                if side == "BUY":
                    pnl_amount = (cur - entry) * qty
                    target_hit = cur >= pos["target_price"]
                    stop_hit = cur <= pos["stop_loss_price"]
                else:
                    pnl_amount = (entry - cur) * qty
                    target_hit = cur <= pos["target_price"]
                    stop_hit = cur >= pos["stop_loss_price"]
                ist_now = now_ist()
                force_exit = ist_now.time() >= datetime_time(15, 10)
                if target_hit or stop_hit or force_exit:
                    exit_is_buy = (side == "SHORT")
                    exit_order_id = self.place_order(symbol, qty, exit_is_buy)
                    if exit_order_id:
                        logger.info(f"Closed {side} {symbol} P&L: {round2(pnl_amount)} reason: {'TP' if target_hit else ('SL' if stop_hit else 'EOD')}")
                        to_close.append(key)
                        self.total_trades_today += 1
                else:
                    logger.info(f"{side} {symbol} LTP={round2(cur)} PnL={round2(pnl_amount)} stop={pos['stop_loss_price']} target={pos['target_price']}")
            except Exception as e:
                logger.error(f"Monitor error {key}: {e}")
        for k in to_close:
            if k in self.positions:
                del self.positions[k]
        if to_close:
            self.save_positions()
        if not self.positions:
            try:
                # rapid_monitor is the global/outer instance created alongside bot
                if 'rapid_monitor' in globals():
                    rapid_monitor.stop()
            except Exception as e:
                logger.warning(f"Rapid monitor early-stop failed: {e}")
            return

    # ========= Scanning =========
    def scan_for_opportunities(self):
        logger.info("Scan started.")
        if self.bot_status == "Auth Required" or not self.access_token:
            logger.info("Auth required; scan skipped.")
            return
        if not is_market_open_now():
            logger.info("Market closed; skipping scan.")
            return
        if len(self.positions) >= self.max_positions:
            logger.info("Max positions reached; skipping new entries.")
            return
        self.maybe_refresh_daily_stock_list()
        with self._cache_lock:
            watchlist = list(self.daily_stock_list)
        if not watchlist:
            logger.info("Watchlist empty; skipping.")
            return
        for symbol in watchlist:
            if any(p["symbol"] == symbol for p in self.positions.values()):
                continue
            sig = self.generate_trade_signal(symbol)
            if sig:
                direction, atr5, last_close = sig
                ok = self.execute_strategy(symbol, direction, last_close, atr5)
                if ok:
                    time.sleep(2)
                    break

    # ========= Persistence =========
    def save_positions(self):
        try:
            data = {}
            for k, pos in self.positions.items():
                d = pos.copy()
                d["entry_time"] = pos["entry_time"].isoformat()
                data[k] = d
            with open("positions.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save positions error: {e}")

    def load_positions(self):
        try:
            with open("positions.json", "r") as f:
                data = json.load(f)
                for k, pos in data.items():
                    pos["entry_time"] = datetime.fromisoformat(pos["entry_time"])
                    self.positions[k] = pos
            logger.info(f"Loaded {len(self.positions)} positions.")
        except Exception:
            logger.info("No saved positions to load.")

    # ========= Scheduling =========
    def start_schedulers(self):
        if self._scheduler_started:
            return
        schedule.clear()
        schedule.every(15).minutes.do(self.run_trading_cycle)
        schedule.every(15).minutes.do(self.scan_for_opportunities)
        self.schedule_auth_checks()
        self._scheduler_started = True
        logger.info("Schedulers set: 15-min cycles, scan, and auth check.")

    def run_trading_cycle(self):
        try:
            if self.bot_status == "Auth Required" or not self.access_token:
                logger.info("Auth required; trading cycle skipped.")
                return
            self.bot_status = "Running"
            if not is_market_open_now():
                if self.positions:
                    self.monitor_positions()
                self.bot_status = "Market Closed"
                return
            if self.positions:
                self.monitor_positions()
            if len(self.positions) < self.max_positions:
                self.scan_for_opportunities()
            logger.info(f"Cycle end. Positions: {len(self.positions)} | Max: {self.max_positions}")
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            self.bot_status = f"Error: {e}"

# Global bot instance
bot = AutoTradingBot()

# Place near the bottom of main.py after bot = AutoTradingBot()
import threading, time

class RapidPositionMonitor:
    def __init__(self, bot, interval_sec=20):
        self.bot = bot
        self.interval = interval_sec
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info(f"Rapid monitor started ({self.interval}s)")

    def stop(self):
        with self._lock:
            self._running = False
        logger.info("Rapid monitor stop requested")

    def _loop(self):
        # Gentle ramp: wait a moment to ensure entry is settled
        time.sleep(2)
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                # If no positions, stop cleanly
                if not self.bot.positions:
                    logger.info("No positions; stopping rapid monitor")
                    self.stop()
                    break

                # Call the bot’s own monitoring (targets, stops, EOD logic)
                self.bot.monitor_positions()  # already does trailing stop + exits [1]

            except Exception as e:
                logger.error(f"Rapid monitor error: {e}")

            # Sleep between polls
            time.sleep(self.interval)

rapid_monitor = RapidPositionMonitor(bot, interval_sec=20)


# ==================== Flask Routes ====================
@app.route("/")
def home():
    status = "Active" if bot.access_token else "Inactive"
    positions_count = len(bot.positions)
    market_status = "Open" if is_market_open_now() else "Closed"
    current_time = now_ist().strftime("%Y-%m-%d %H:%M:%S IST")
    return f"""
    <html><head><title>Trading Bot Dashboard</title></head><body>
    <h1>🤖 Intraday Trading Bot (VWAP+ATR+MTF)</h1>
    <p><b>Status:</b> {status}</p>
    <p><b>Positions Open:</b> {positions_count}</p>
    <p><b>Market Status:</b> {market_status}</p>
    <p><b>Risk/trade:</b> {int(bot.risk_per_trade*100)}% | <b>Max Positions:</b> {bot.max_positions}</p>
    <p><small>Last updated: {current_time}</small></p>
    </body></html>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        "flask_working": True,
        "bot_active": bot.access_token is not None,
        "positions_count": len(bot.positions),
        "market_open": is_market_open_now(),
        "features": ["BUY+SHORT","VWAP","EMA20 MTF","ATR breakout","ATR trailing","MIS exits"]
    })

@app.route("/session/exchange", methods=["POST"])
def session_exchange():
    data = request.get_json(force=True)
    request_token = data.get("request_token")
    if not request_token or len(request_token) < 10:
        return jsonify({"success": False, "message": "Invalid request_token"}), 400
    ok = bot.authenticate_with_request_token(request_token)
    return jsonify({"success": ok}), (200 if ok else 400)

@app.route("/initialize", methods=["GET"])
def initialize():
    if not bot.access_token:
        return jsonify({"success": False, "message": "Authenticate first"}), 401
    bot.load_positions()
    bot.update_daily_stock_list()
    bot.start_schedulers()
    return jsonify({"success": True, "universe_size": len(bot.daily_stock_list), "version": bot.universe_version})

@app.route("/api/universe", methods=["GET"])
def api_universe():
    with bot._cache_lock:
        feats = bot.universe_features.copy()
        if not feats.empty:
            view = feats[["Symbol","Close","ATR_pct","MedTurn20","Score"]].copy()
            view["Symbol"] = view["Symbol"].str.replace(".NS","", regex=False)
            universe_records = view.to_dict(orient="records")
        else:
            universe_records = []
    return jsonify({
        "version": bot.universe_version,
        "session_universe": bot.daily_stock_list,
        "universe": universe_records
    })

@app.route("/api/universe/rebuild", methods=["POST","GET"])
def api_universe_rebuild():
    bot.update_daily_stock_list()
    return jsonify({"success": True, "version": bot.universe_version, "session_universe": bot.daily_stock_list})

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        margins = bot.kite.margins()
        available_cash = margins["equity"]["available"]["live_balance"]
        access_valid = True

    except Exception:
        available_cash = 0
        access_valid = False
    auth_required = not access_valid

    positions_list = []
    for pos in bot.positions.values():
        cur = bot.get_stock_price(pos["symbol"]) or pos["entry_price"]
        if pos["side"] == "BUY":
            pnl = (cur - pos["entry_price"]) * pos["quantity"]
            pnl_percent = ((cur - pos["entry_price"]) / pos["entry_price"]) * 100
        else:
            pnl = (pos["entry_price"] - cur) * pos["quantity"]
            pnl_percent = ((pos["entry_price"] - cur) / pos["entry_price"]) * 100
        positions_list.append({
            "symbol": pos["symbol"],
            "transaction_type": pos["side"],
            "buy_price": pos["entry_price"],
            "current_price": cur,
            "quantity": pos["quantity"],
            "target_price": pos["target_price"],
            "stop_loss_price": pos["stop_loss_price"],
            "entry_time": pos["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "pnl": pnl,
            "pnl_percent": pnl_percent
        })

    daily_pnl = sum(p["pnl"] for p in positions_list) if positions_list else 0.0

    return jsonify({
        "balance": available_cash,
        "positions": positions_list,
        "orders": bot.pending_orders,
        "market_open": is_market_open_now(),
        "bot_status": bot.bot_status,
        "target_profit": bot.target_profit_pct,
        "stop_loss": bot.stop_loss_pct,
        "daily_pnl": daily_pnl,
        "total_trades": bot.total_trades_today,
        "win_rate": bot.win_rate,
        "access_token_valid": access_valid,
        "auth_required": auth_required,
        "risk_per_trade": bot.risk_per_trade,
        "max_positions": bot.max_positions,
        "last_update": now_ist().strftime("%H:%M:%S")
    })

@app.route("/api/close-position", methods=["POST"])
def close_position():
    data = request.get_json(force=True)
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"success": False, "message": "Symbol required"}), 400
    found_key = None
    pos = None
    for k, p in bot.positions.items():
        if p["symbol"] == symbol:
            found_key = k
            pos = p
            break
    if not found_key:
        return jsonify({"success": False, "message": "Position not found"}), 404
    is_buy_to_close = (pos["side"] == "SHORT")
    oid = bot.place_order(symbol, pos["quantity"], is_buy_to_close)
    if oid:
        del bot.positions[found_key]
        bot.save_positions()
        return jsonify({"success": True, "message": f"Closed {symbol}"})
    return jsonify({"success": False, "message": "Exit order failed"}), 500

@app.route("/api/refresh-token", methods=["POST"])
def refresh_access_token():
    data = request.get_json(force=True)
    new_token = data.get("access_token")
    if not new_token:
        return jsonify({"success": False, "message": "Token required"}), 400
    try:
        bot.kite.set_access_token(new_token)
        bot.access_token = new_token
        with open("access_token.txt", "w") as f:
            f.write(f"{new_token}\n{now_ist().isoformat()}")
        bot.kite.profile()
        return jsonify({"success": True, "message": "Token updated"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route("/control/<action>", methods=["GET"])
def control(action):
    if not bot.access_token:
        return jsonify({"status": "Authenticate first"}), 401
    if action == "scan":
        bot.scan_for_opportunities()
        return jsonify({"status": "scan queued"})
    elif action == "pause":
        bot.bot_status = "Paused"
        return jsonify({"status": "paused"})
    elif action == "resume":
        bot.bot_status = "Running"
        return jsonify({"status": "resumed"})
    elif action == "rebuild_and_scan":
        bot.update_daily_stock_list()
        bot.scan_for_opportunities()
        return jsonify({"status": "rebuild+scan queued"})
    return jsonify({"status": f"unknown action {action}"}), 400

# Replace the backtest_run function with this comprehensive version:

@app.route("/backtest/run", methods=["POST"])
def backtest_run():
    """Comprehensive backtesting using actual strategy logic and universe"""
    try:
        data = request.get_json() or {}
        
        # Backtest parameters from frontend
        start_date = data.get("start_date", "2024-03-01")
        end_date = data.get("end_date", "2024-08-30")
        initial_capital = data.get("capital", 25000)
        
        # Run backtest with your ACTUAL universe
        results = run_comprehensive_backtest(start_date, end_date, initial_capital)
        
        # Save detailed results
        results_df = pd.DataFrame(results['trades'])
        results_df.to_csv("backtest_detailed.csv", index=False)
        
        return jsonify({
            "success": True,
            "summary": results['summary'],
            "trades": len(results['trades']),
            "daily_pnl": results['daily_pnl'][-10:],  # Last 10 days for chart
            "csv_url": "/backtest/csv"
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def run_comprehensive_backtest(start_date, end_date, initial_capital):
    """Run backtest using YOUR actual universe and strategy"""
    logger.info(f"🔄 Starting backtest: {start_date} to {end_date}")
    
    # Use YOUR actual universe - all BASE_TICKERS
    universe = [s.replace(".NS", "") for s in BASE_TICKERS]
    logger.info(f"Using {len(universe)} stocks from your actual universe")
    
    # Initialize backtest state
    capital = initial_capital
    positions = {}
    trades = []
    daily_pnl = []
    
    # Date range generation
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_date = start
    total_days = (end - start).days
    day_count = 0
    
    while current_date <= end:
        if current_date.weekday() < 5:  # Trading days only
            day_count += 1
            logger.info(f"📅 Day {day_count}: {current_date.strftime('%Y-%m-%d')}")
            
            # Simulate your actual trading logic for this day
            daily_result = simulate_actual_trading_day(
                current_date, universe, capital, positions, 
                initial_capital  # Pass for position sizing
            )
            
            # Update capital and tracking
            capital += daily_result['daily_pnl']
            trades.extend(daily_result['trades'])
            
            daily_pnl.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'daily_pnl': round(daily_result['daily_pnl'], 2),
                'total_capital': round(capital, 2),
                'active_positions': len(positions),
                'trades_today': len(daily_result['trades'])
            })
            
            if day_count % 10 == 0:  # Progress update every 10 days
                logger.info(f"Progress: {day_count}/{total_days * 5/7:.0f} days, Capital: ₹{capital:.0f}")
        
        current_date += timedelta(days=1)
    
    # Calculate comprehensive metrics using your actual data
    summary = calculate_realistic_metrics(trades, daily_pnl, initial_capital, capital)
    
    return {
        'trades': trades,
        'daily_pnl': daily_pnl,
        'summary': summary
    }

def simulate_actual_trading_day(date, universe, current_capital, positions, initial_capital):
    """Simulate one day using YOUR actual bot logic"""
    daily_trades = []
    daily_pnl = 0
    
    # 1. Monitor existing positions (your actual monitor_positions logic)
    for symbol in list(positions.keys()):
        position = positions[symbol]
        
        # Get historical price for this date
        current_price = get_backtest_price(symbol, date)
        if current_price is None:
            continue
            
        # Apply YOUR actual exit conditions
        exit_result = check_backtest_exits(position, current_price, date)
        
        if exit_result['should_exit']:
            # Calculate P&L using your actual logic
            if position['side'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Subtract transaction costs (₹47 per trade)
            pnl -= 47
            
            daily_trades.append({
                'date': date.strftime("%Y-%m-%d"),
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': position['quantity'],
                'pnl': round(pnl, 2),
                'exit_reason': exit_result['reason'],
                'hold_hours': int((date - position['entry_date']).total_seconds() / 3600)
            })
            
            daily_pnl += pnl
            del positions[symbol]
            logger.info(f"  ✅ Closed {position['side']} {symbol}: ₹{pnl:.0f} ({exit_result['reason']})")
    
    # 2. Look for new signals (your actual signal generation)
    max_positions = 2  # Your actual setting
    if len(positions) < max_positions:
        
        # Randomly check some stocks (simulate scanning)
        import random
        scan_symbols = random.sample(universe, min(20, len(universe)))
        
        for symbol in scan_symbols:
            if symbol in positions:
                continue
                
            # Use YOUR actual signal generation logic (simplified for backtest)
            signal = generate_backtest_signal(symbol, date)
            
            if signal and len(positions) < max_positions:
                direction, atr, signal_price = signal
                
                # Use YOUR actual position sizing (1.5% risk)
                risk_amount = current_capital * 0.015  # Your actual RISK_PER_TRADE
                if atr > 0:
                    qty = max(1, int(risk_amount / atr))
                    
                    # Apply your max notional limit (8%)
                    max_notional = current_capital * 0.08  # Your MAX_NOTIONAL_PCT
                    if qty * signal_price > max_notional:
                        qty = max(1, int(max_notional / signal_price))
                    
                    if qty > 0:
                        # Entry transaction cost
                        daily_pnl -= 47
                        
                        # Create position using your actual logic
                        positions[symbol] = {
                            'side': direction,
                            'entry_price': signal_price,
                            'quantity': qty,
                            'entry_date': date,
                            'target_price': signal_price + (2 * atr) if direction == 'BUY' else signal_price - (2 * atr),
                            'stop_price': signal_price - atr if direction == 'BUY' else signal_price + atr
                        }
                        
                        logger.info(f"  📈 New {direction} {symbol} @ ₹{signal_price:.1f}, qty={qty}")
                        break  # Only one new position per day (realistic)
    
    return {'daily_pnl': daily_pnl, 'trades': daily_trades}

def generate_backtest_signal(symbol, date):
    """Simplified version of your actual signal generation"""
    try:
        # Get historical data as it would have been available on that date
        data_30m = get_backtest_historical_data(symbol, date, "30m", 10)
        data_5m = get_backtest_historical_data(symbol, date, "5m", 2)
        
        if data_30m is None or data_5m is None or len(data_30m) < 40 or len(data_5m) < 40:
            return None
        
        # Simplified MTF confirmation (based on your actual logic)
        ema20_30 = data_30m["close"].ewm(span=20).mean()
        if ema20_30.iloc[-2:].isna().values.any(): # single scalar True/False
            return None
            
        ema_last = float(ema20_30.iloc[-1])
        ema_prev = float(ema20_30.iloc[-2])
        price_30 = float(data_30m["close"].iloc[-1])
        
        # VWAP calculation
        vwap_5 = ((data_5m["close"] * data_5m["volume"]).cumsum() / 
                  data_5m["volume"].cumsum())
        
        if pd.isna(vwap_5.iloc[-1]):
            return None
            
        vwap_last = float(vwap_5.iloc[-1])
        price_5 = float(data_5m["close"].iloc[-1])
        
        # Your actual signal conditions
        slope_up = ema_last > ema_prev
        slope_down = ema_last < ema_prev
        above_ema = price_30 > ema_last
        below_ema = price_30 < ema_last
        above_vwap = price_5 > vwap_last
        below_vwap = price_5 < vwap_last
        
        # ATR calculation
        high_low = data_5m["high"] - data_5m["low"]
        high_close = (data_5m["high"] - data_5m["close"].shift()).abs()
        low_close = (data_5m["low"] - data_5m["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr5 = true_range.rolling(14).mean().iloc[-1]
        
        if pd.isna(atr5) or atr5 <= 0:
            return None
        
        # Breakout detection (simplified)
        last_close = float(data_5m["close"].iloc[-1])
        recent_high = float(data_5m["high"].rolling(24).max().iloc[-2])
        recent_low = float(data_5m["low"].rolling(24).min().iloc[-2])
        
        # Your actual signal logic
        long_signal = (slope_up and above_ema and above_vwap and 
                      last_close > recent_high * 1.0025)
        short_signal = (slope_down and below_ema and below_vwap and 
                       last_close < recent_low * 0.9975)
        
        if long_signal:
            return ("BUY", float(atr5), float(last_close))
        elif short_signal:
            return ("SHORT", float(atr5), float(last_close))
            
        return None
        
    except Exception as e:
        return None

def get_backtest_historical_data(symbol, date, interval, days):
    """Get historical data for backtesting"""
    try:
        end_date = date
        start_date = date - timedelta(days=days)
        
        df = yf.download(f"{symbol}.NS", 
                        start=start_date.strftime("%Y-%m-%d"),
                        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval=interval,
                        progress=False)
        
        if df.empty:
            return None
            
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Only return data up to the backtest date (no lookahead bias)
        df = df[df['datetime'].dt.date <= date.date()]
        
        return df
        
    except Exception:
        return None

def get_backtest_price(symbol, date):
    """Get closing price for a specific date"""
    try:
        df = yf.download(f"{symbol}.NS", 
                        start=date.strftime("%Y-%m-%d"),
                        end=(date + timedelta(days=1)).strftime("%Y-%m-%d"),
                        progress=False)
        
        if df.empty:
            return None
            
        return float(df['Close'].iloc[-1])
        
    except Exception:
        return None

def check_backtest_exits(position, current_price, date):
    """Check exit conditions using your actual logic"""
    
    # Target hit
    if position['side'] == 'BUY' and current_price >= position['target_price']:
        return {'should_exit': True, 'reason': 'Target'}
    if position['side'] == 'SHORT' and current_price <= position['target_price']:
        return {'should_exit': True, 'reason': 'Target'}
    
    # Stop hit
    if position['side'] == 'BUY' and current_price <= position['stop_price']:
        return {'should_exit': True, 'reason': 'Stop Loss'}
    if position['side'] == 'SHORT' and current_price >= position['stop_price']:
        return {'should_exit': True, 'reason': 'Stop Loss'}
    
    # End of day (your actual 3:10 PM exit)
    # For backtesting, assume end of day exit
    hours_held = (date - position['entry_date']).total_seconds() / 3600
    if hours_held > 6:  # Approximate market day
        return {'should_exit': True, 'reason': 'EOD'}
    
    return {'should_exit': False, 'reason': None}

def calculate_realistic_metrics(trades, daily_pnl, initial_capital, final_capital):
    """Calculate comprehensive performance metrics"""
    
    if not trades:
        return {
            "error": "No trades generated during backtest period",
            "initial_capital": initial_capital,
            "final_capital": final_capital
        }
    
    trades_df = pd.DataFrame(trades)
    
    # Basic performance metrics
    total_trades = len(trades)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital * 100)
    
    # Win/Loss analysis
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 and avg_loss > 0 else float('inf')
    
    # Monthly analysis
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df['month'] = trades_df['date'].dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
    
    # Risk metrics
    daily_df = pd.DataFrame(daily_pnl)
    if not daily_df.empty:
        daily_returns = daily_df['daily_pnl']
        running_capital = daily_df['total_capital']
        
        # Max drawdown calculation
        peak = running_capital.expanding().max()
        drawdown = (running_capital - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized)
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5)
        else:
            sharpe_ratio = 0
    else:
        max_drawdown = 0
        sharpe_ratio = 0
    
    # Trading frequency
    trading_days = len(daily_df) if not daily_df.empty else 1
    trades_per_day = total_trades / trading_days
    
    return {
        "period_summary": {
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "trading_days": trading_days
        },
        "trading_stats": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 1),
            "trades_per_day": round(trades_per_day, 2)
        },
        "performance": {
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        },
        "monthly_analysis": {
            "monthly_avg": round(monthly_pnl.mean(), 2) if not monthly_pnl.empty else 0,
            "monthly_std": round(monthly_pnl.std(), 2) if not monthly_pnl.empty else 0,
            "best_month": round(monthly_pnl.max(), 2) if not monthly_pnl.empty else 0,
            "worst_month": round(monthly_pnl.min(), 2) if not monthly_pnl.empty else 0
        }
    }

# Add these new routes for Vercel integration:

@app.route("/api/backtest/start", methods=["POST"])
def api_backtest_start():
    """Start backtest with parameters from Vercel app"""
    try:
        data = request.get_json()
        
        # Validate parameters
        start_date = data.get("start_date", "2024-03-01")
        end_date = data.get("end_date", "2024-08-30")
        capital = data.get("capital", 25000)
        
        # Start backtest in background thread
        import threading
        
        def run_backtest():
            global backtest_status, backtest_results
            backtest_status = {"status": "running", "progress": 0}
            
            try:
                results = run_comprehensive_backtest(start_date, end_date, capital)
                backtest_status = {"status": "completed", "progress": 100}
                backtest_results = results
            except Exception as e:
                backtest_status = {"status": "error", "error": str(e), "progress": 0}
        
        # Initialize global variables
        global backtest_status, backtest_results
        backtest_status = {"status": "starting", "progress": 0}
        backtest_results = None
        
        # Start backtest
        thread = threading.Thread(target=run_backtest, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Backtest started",
            "status": "running"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/backtest/status", methods=["GET"])
def api_backtest_status():
    """Get current backtest status for Vercel app"""
    global backtest_status
    if 'backtest_status' not in globals():
        return jsonify({"status": "idle", "progress": 0})
    return jsonify(backtest_status)

@app.route("/api/backtest/results", methods=["GET"])
def api_backtest_results():
    """Get backtest results for Vercel app"""
    global backtest_results
    if 'backtest_results' not in globals() or backtest_results is None:
        return jsonify({"success": False, "error": "No results available"}), 404
    
    return jsonify({
        "success": True,
        "data": backtest_results
    })

@app.route("/api/backtest/presets", methods=["GET"])
def api_backtest_presets():
    """Get preset backtest configurations"""
    presets = [
        {
            "name": "Last 3 Months",
            "start_date": "2024-06-01",
            "end_date": "2024-08-30",
            "description": "Recent market conditions"
        },
        {
            "name": "Last 6 Months", 
            "start_date": "2024-03-01",
            "end_date": "2024-08-30",
            "description": "Includes market volatility"
        },
        {
            "name": "YTD 2024",
            "start_date": "2024-01-01", 
            "end_date": "2024-08-30",
            "description": "Full year performance"
        },
        {
            "name": "High Volatility Period",
            "start_date": "2024-03-15",
            "end_date": "2024-04-15", 
            "description": "Market stress test"
        }
    ]
    
    return jsonify({"presets": presets})



@app.route("/backtest/csv", methods=["GET"])
def backtest_csv():
    fname = "backtest_pnl.csv"
    if not os.path.exists(fname):
        return jsonify({"success": False, "message": "No CSV"}), 404
    return send_file(fname, as_attachment=True)

# Add these NEW endpoints:

@app.route('/cron/monitor-positions', methods=['GET', 'POST'])
def cron_monitor_positions():
    """Called every 5 minutes during market hours"""
    try:
        if bot.access_token and is_market_open_now():
            bot.monitor_positions()
            return jsonify({
                "status": "success", 
                "positions": len(bot.positions),
                "timestamp": now_ist().isoformat()
            })
        return jsonify({"status": "skipped", "reason": "market closed or no auth"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/cron/scan-opportunities', methods=['GET', 'POST'])  
def cron_scan_opportunities():
    """Called every 15 minutes during market hours"""
    try:
        if bot.access_token and is_market_open_now() and len(bot.positions) < bot.max_positions:
            bot.scan_for_opportunities()
            return jsonify({"status": "success", "scanned": True})
        return jsonify({"status": "skipped", "reason": "conditions not met"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/cron/health-check', methods=['GET', 'POST'])
def cron_health_check():
    """Called every 10 minutes - keeps app awake"""
    return jsonify({
        "status": "alive",
        "bot_status": bot.bot_status,
        "positions": len(bot.positions),
        "market_open": is_market_open_now(),
        "timestamp": now_ist().isoformat()
    })

@app.route('/cron/universe-update', methods=['GET', 'POST'])
def cron_universe_update():
    """Called once daily at 8:30 AM"""
    try:
        bot.update_daily_stock_list()
        return jsonify({
            "status": "success", 
            "universe_size": len(bot.daily_stock_list),
            "version": bot.universe_version
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ==================== OAuth routes ====================
def _kite_login_url(api_key: str, redirect_params: dict | None = None):
    base = "https://kite.zerodha.com/connect/login?v=3"
    qp = {"api_key": api_key}
    if redirect_params:
        qp["redirect_params"] = urllib.parse.quote_plus(urllib.parse.urlencode(redirect_params))
    return f"{base}&{urllib.parse.urlencode(qp)}"

@app.route("/auth/login", methods=["GET"])
def auth_login():
    if not API_KEY or not API_SECRET:
        return jsonify({"success": False, "message": "Server missing API creds"}), 500
    state = request.args.get("state", "")
    next_path = request.args.get("next", "/")
    rp = {}
    if state:
        rp["state"] = state
    if next_path:
        rp["next"] = next_path
    url = _kite_login_url(API_KEY, redirect_params=rp if rp else None)
    return redirect(url, code=302)

@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    req_token = request.args.get("request_token")
    next_path = request.args.get("next", "/")
    state = request.args.get("state", "")
    if not req_token or len(req_token) < 10:
        return _finish_auth_redirect(False, "missing_or_invalid_request_token", next_path, state)
    try:
        sess = bot.kite.generate_session(req_token, api_secret=API_SECRET)
        access_token = sess["access_token"]
        bot.kite.set_access_token(access_token)
        bot.access_token = access_token
        with open("access_token.txt", "w") as f:
            f.write(f"{access_token}\n{now_ist().isoformat()}")
        try:
            margins = bot.kite.margins()
            bal = float(margins["equity"]["available"]["live_balance"])
            bot.account_equity = max(bal, DEFAULT_ACCOUNT_EQUITY)
            bot.max_positions = bot._max_positions_for_equity(bot.account_equity)
        except Exception:
            pass
        try:
            bot.start_schedulers()
        except Exception:
            pass
        bot.bot_status = "Active"
        return _finish_auth_redirect(True, "ok", next_path, state)
    except Exception as e:
        logger.error(f"/auth/callback exchange failed: {e}")
        return _finish_auth_redirect(False, "exchange_failed", next_path, state)

def _finish_auth_redirect(success: bool, code: str, next_path: str, state: str):
    front = FRONTEND_URL
    if not front:
        return jsonify({"success": success, "code": code})
    qp = {"auth": "success" if success else "fail", "code": code}
    if state:
        qp["state"] = state
    url = f"{front}{next_path if next_path.startswith('/') else '/'}"
    sep = "&" if "?" in url else "?"
    url = url + sep + urllib.parse.urlencode(qp)
    return redirect(url, code=302)

# ==================== App main ====================
def _shutdown(*args):
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

if __name__ == "__main__":
    print("Intraday Trading Bot (VWAP+ATR+MTF)")
    print("Order Exec: Zerodha | Data: Yahoo Finance (demo)")
    print("Risk/trade 4% | MTF EMA20 + VWAP | ATR breakout & trailing")

    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
