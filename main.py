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
from requests.adapters import HTTPAdapter, Retry

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from kiteconnect import KiteConnect

#==================== Config & Globals ====================
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = datetime_time(9, 15)
MARKET_CLOSE = datetime_time(15, 30)

API_KEY = os.environ.get('KITE_API_KEY')
API_SECRET = os.environ.get('KITE_API_SECRET')

if not API_KEY or not API_SECRET:
    print("ERROR: Missing KITE_API_KEY or KITE_API_SECRET in environment.")
raise SystemExit(1)

#Small-capital friendly risk defaults
DEFAULT_ACCOUNT_EQUITY = float(os.environ.get("ACCOUNT_EQUITY", "10000")) # demo only; real balance via margins
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01")) # 1% per trade
MAX_POSITIONS_10K = int(os.environ.get("MAX_POS_10K", "2"))
MAX_POSITIONS_20K = int(os.environ.get("MAX_POS_20K", "3"))
MAX_POSITIONS_30K = int(os.environ.get("MAX_POS_30K", "3"))
MAX_NOTIONAL_PCT = float(os.environ.get("MAX_NOTIONAL_PCT", "0.15")) # cap notional/trade

UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "40")) # daily universe shortlist size
HYST_ADD_RANK = int(os.environ.get("HYST_ADD_RANK", "30"))
HYST_DROP_RANK = int(os.environ.get("HYST_DROP_RANK", "50"))

BACKTEST_YEARS = int(os.environ.get("BACKTEST_YEARS", "1"))

BASE_TICKERS = [
"RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","KOTAKBANK.NS","SBIN.NS",
"BHARTIARTL.NS","LT.NS","AXISBANK.NS","ITC.NS","HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS",
"ULTRACEMCO.NS","SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ASIANPAINT.NS","HCLTECH.NS","NESTLEIND.NS",
"M&M.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","ADANIENT.NS","ADANIPORTS.NS","JSWSTEEL.NS","TATASTEEL.NS",
"COALINDIA.NS","DIVISLAB.NS","TECHM.NS","LTIM.NS","BRITANNIA.NS","BPCL.NS","EICHERMOT.NS","HDFCLIFE.NS",
"DRREDDY.NS","SBILIFE.NS","GRASIM.NS","HINDALCO.NS","INDUSINDBK.NS","BAJAJFINSV.NS","TATAMOTORS.NS","HEROMOTOCO.NS"
]

app = Flask(name)
CORS(app, origins=['*'])

#Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TradingBot")

#Web keep-alive
STOP_EVENT = threading.Event()

#==================== Helpers ====================
def now_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def is_market_open_now():
    t = now_ist()
    if t.weekday() >= 5:
    return False
    return (t.hour > 9 or (t.hour == 9 and t.minute >= 15)) and (t.hour < 15 or (t.hour == 15 and t.minute <= 30))

def _make_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=, allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def round2(x):
try:
    return round(float(x), 2)
    except Exception:
return x

#==================== Bot Class ====================
class AutoTradingBot:
def init(self):
self.kite = KiteConnect(api_key=API_KEY)
self.access_token = None

text
    # State
    self.positions = {}  # key -> dict(symbol, side, entry_price, qty, target_price, stop_loss_price, entry_time, order_id)
    self.pending_orders = []
    self.bot_status = "Initializing"
    self.total_trades_today = 0
    self.win_rate = 0.0

    # Strategy params
    self.target_profit_pct = 1.0   # default target for display; execution uses ATR-based targets/trailing
    self.stop_loss_pct = 0.5       # display only; execution uses ATR-based SL
    self.trailing_start_R = 0.5    # start trailing after 0.5R
    self.trailing_atr_mult = 1.0   # trail by 1x 5m ATR

    # Risk
    self.account_equity = DEFAULT_ACCOUNT_EQUITY
    self.risk_per_trade = RISK_PER_TRADE
    self.max_positions = self._max_positions_for_equity(self.account_equity)
    self.max_notional_pct = MAX_NOTIONAL_PCT

    # Universe
    self._cache_lock = threading.Lock()
    self._last_cache_update = None
    self.daily_stock_list = []
    self.universe_version = None
    self.universe_features = pd.DataFrame()  # snapshot with ATR%, Turnover, Score

    # Threads / scheduling
    self._scheduler_started = False

    logger.info("Bot initialized.")

# ========= Session / Auth =========
def authenticate_with_request_token(self, request_token: str):
    try:
        sess = self.kite.generate_session(request_token, api_secret=API_SECRET)
        self.access_token = sess["access_token"]
        self.kite.set_access_token(self.access_token)
        self.bot_status = "Active"
        # Persist
        ts = now_ist().replace(microsecond=0).isoformat()
        with open("access_token.txt", "w") as f:
            f.write(f"{self.access_token}\n{ts}")
        logger.info("Authentication successful; token saved.")
        # Test profile
        try:
            profile = self.kite.profile()
            margins = self.kite.margins()
            bal = margins["equity"]["available"]["live_balance"]
            logger.info(f"Welcome: {profile.get('user_name','?')} | Balance: ₹{bal:,.2f}")
            # update equity heuristic
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
            lines = f.read().strip().split("\n")
            token = lines
            self.kite.set_access_token(token)
            self.access_token = token
            self.bot_status = "Active"
            return True
    except Exception:
        return False

# ========= Universe Builder =========
def _fetch_eod_batch(self, tickers, period="12mo"):
    data = yf.download(tickers, period=period, interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
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

def _compute_features(self, df):
    # df: Date, Open, High, Low, Close, Adj Close, Volume, Symbol
    def featurize(g):
        g = g.sort_values("Date").copy()
        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift(1)).abs()
        tr3 = (g["Low"] - g["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["ATR"] = tr.rolling(20).mean()
        g["ATR_pct"] = (g["ATR"] / g["Close"]) * 100.0
        g["Turnover"] = g["Close"] * g["Volume"]
        g["MedTurn20"] = g["Turnover"].rolling(20).median()
        g["Ret20"] = g["Close"].pct_change(20)
        g["GapRisk"] = (g["Open"] - g["Close"].shift(1)).abs().rolling(20).std()
        return g
    df = df.groupby("Symbol", group_keys=False).apply(featurize)
    last = df.sort_values(["Symbol","Date"]).groupby("Symbol").tail(1)
    # ranks
    last["ATRpct_rank"] = last["ATR_pct"].rank(pct=True)
    last["Turn_rank"] = last["MedTurn20"].rank(pct=True)
    last["Mom_rank"] = last["Ret20"].rank(pct=True)
    last["Gap_rank"] = last["GapRisk"].rank(pct=True)
    last["Score"] = 0.45*last["ATRpct_rank"] + 0.40*last["Turn_rank"] + 0.25*last["Mom_rank"] - 0.20*last["Gap_rank"]
    return last

def update_daily_stock_list(self):
    try:
        with self._cache_lock:
            logger.info("Building dynamic universe...")
            eod = self._fetch_eod_batch(BASE_TICKERS, period="12mo")
            if eod.empty:
                logger.warning("Universe build failed: no EOD data.")
                return
            feats = self._compute_features(eod)
            feats = feats[(feats["MedTurn20"] > 2e7) & (feats["Close"] >= 50)]
            feats = feats.sort_values("Score", ascending=False).head(UNIVERSE_SIZE).reset_index(drop=True)
            # Session hysteresis (simple)
            prev = set(self.daily_stock_list)
            ranked = list(feats["Symbol"].values)
            add = [s for i, s in enumerate(ranked) if (s not in prev) and (i < HYST_ADD_RANK)]
            keep = [s for i, s in enumerate(ranked) if (s in prev) or (i < HYST_DROP_RANK)]
            session_list = list(dict.fromkeys(keep + add))[:UNIVERSE_SIZE]
            # Convert to NSE trading symbols (drop .NS)
            self.daily_stock_list = [s.replace(".NS","") for s in session_list]
            self.universe_features = feats.copy()
            self.universe_version = now_ist().strftime("%Y-%m-%d")
            self._last_cache_update = now_ist()
            logger.info(f"Universe built with {len(self.daily_stock_list)} symbols.")
    except Exception as e:
        logger.error(f"Failed to build universe: {e}")

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

# ========= Data Fetch / Indicators =========
def fetch_bars(self, symbol, interval='5m', days=2):
    try:
        yf_symbol = f"{symbol}.NS"
        df = yf.download(yf_symbol, period=f"{days}d", interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            return None
        df = df.rename(columns=str.lower).reset_index()
        # ensure columns: open, high, low, close, volume
        return df
    except Exception:
        return None

def compute_vwap(self, df):
    # df with close, volume
    pv = (df["close"] * df["volume"]).cumsum()
    vv = (df["volume"]).cumsum().replace(0, np.nan)
    return pv / vv

def ema(self, series, n):
    return series.ewm(span=n, adjust=False).mean()

def get_stock_price(self, symbol):
    # Prefer broker LTP if authenticated; else yfinance fast_info
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

# ========= Multi-timeframe confirmation =========
def mtf_confirmation(self, symbol):
    data_30 = self.fetch_bars(symbol, interval='30m', days=10)
    data_5 = self.fetch_bars(symbol, interval='5m', days=2)
    if data_30 is None or data_5 is None or len(data_30) < 40 or len(data_5) < 40:
        return None
    ema20_30 = self.ema(data_30["close"], 20)
    slope_up = ema20_30.iloc[-1] > ema20_30.iloc[-2]
    slope_down = ema20_30.iloc[-1] < ema20_30.iloc[-2]
    price_30 = data_30["close"].iloc[-1]
    above_ema_30 = price_30 > ema20_30.iloc[-1]
    below_ema_30 = price_30 < ema20_30.iloc[-1]

    vwap_5 = self.compute_vwap(data_5)
    price_5 = data_5["close"].iloc[-1]
    above_vwap = price_5 > vwap_5.iloc[-1]
    below_vwap = price_5 < vwap_5.iloc[-1]

    return {
        "long_ok": slope_up and above_ema_30 and above_vwap,
        "short_ok": slope_down and below_ema_30 and below_vwap,
        "data_5": data_5
    }

# ========= Signal Generation (both directions) =========
def generate_trade_signal(self, symbol):
    mtf = self.mtf_confirmation(symbol)
    if mtf is None:
        return None
    data_5 = mtf["data_5"]
    # 5m ATR(14)
    tr = pd.DataFrame({
        "hl": data_5["high"] - data_5["low"],
        "hc": (data_5["high"] - data_5["close"].shift()).abs(),
        "lc": (data_5["low"] - data_5["close"].shift()).abs(),
    }).max(axis=1)
    atr5 = tr.rolling(14).mean().iloc[-1]
    if np.isnan(atr5) or atr5 <= 0:
        return None

    last_close = data_5["close"].iloc[-1]
    prev_24_high = data_5["high"].rolling(24).max().iloc[-2]
    prev_24_low = data_5["low"].rolling(24).min().iloc[-2]

    # Daily ATR% context if available
    daily_atr_pct = 1.0
    try:
        feats = self.universe_features
        row = feats[feats["Symbol"] == f"{symbol}.NS"]
        if not row.empty:
            daily_atr_pct = float(row["ATR_pct"].values)
    except Exception:
        pass
    margin = max(0.0025, 0.0025 * (daily_atr_pct / 1.0))  # ~0.25%

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
    # Risk in rupees
    risk_rupees = self.account_equity * self.risk_per_trade
    if stop_distance <= 0:
        return 0
    qty = int(max(0, np.floor(risk_rupees / stop_distance)))
    # Notional cap
    max_notional = self.account_equity * self.max_notional_pct
    if qty * price > max_notional:
        qty = int(max_notional // price)
    return max(qty, 0)

def place_order(self, symbol, quantity, is_buy):
    try:
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
        ist_now = now_ist()
        if ist_now.time() >= datetime_time(14,45):
            logger.info(f"Skipping late entry on {symbol}")
            return False
        # Balance
        try:
            margins = self.kite.margins()
            bal = margins["equity"]["available"]["live_balance"]
            self.account_equity = max(bal, DEFAULT_ACCOUNT_EQUITY)
            self.max_positions = self._max_positions_for_equity(self.account_equity)
        except Exception:
            pass

        # Initial stop and target
        # Use 1x 5m ATR for stop; target as 1.5R for display; trailing will manage exits
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
            target_price = round2(signal_price + 1.5 * stop_distance)
        else:
            stop_price = round2(signal_price + stop_distance)
            target_price = round2(signal_price - 1.5 * stop_distance)

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

# ========= Trailing stop management =========
def update_trailing_stops(self):
    for key, pos in list(self.positions.items()):
        symbol = pos["symbol"]
        side = pos["side"]
        entry = pos["entry_price"]
        cur = self.get_stock_price(symbol)
        if cur is None:
            continue
        # Get 5m ATR
        data_5 = self.fetch_bars(symbol, interval='5m', days=1)
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

        # compute R from initial stop
        init_stop = pos["stop_loss_price"]
        if side == "BUY":
            R = (pos["target_price"] - entry)  # not exact, but used to trigger trailing after 0.5R
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
    if not self.positions:
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

# ========= Scanning =========
def scan_for_opportunities(self):
    logger.info("Scan started.")
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
        # Skip if already holding symbol
        if any(p["symbol"] == symbol for p in self.positions.values()):
            continue
        sig = self.generate_trade_signal(symbol)
        if sig:
            direction, atr5, last_close = sig
            ok = self.execute_strategy(symbol, direction, last_close, atr5)
            if ok:
                time.sleep(2)
                break  # one trade per scan

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
    # primary cycle every 15 minutes
    schedule.every(15).minutes.do(self.run_trading_cycle)
    # scanner assist (optional separate)
    schedule.every(15).minutes.do(self.scan_for_opportunities)
    self._scheduler_started = True
    logger.info("Schedulers set: 15-min cycles and scan.")

def run_trading_cycle(self):
    try:
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
Global bot instance
bot = AutoTradingBot()

#==================== Keep-alive thread ====================
def keep_alive_ping(interval_seconds=300, timeout_seconds=5, url_env_var="RENDER_PUBLIC_URL"):
session = _make_session()
base_url = os.environ.get(url_env_var)
if not base_url:
logger.warning(f"{url_env_var} not set; keep-alive disabled.")
return
base_url = base_url.rstrip("/")
health_url = f"{base_url}/health"
logger.info(f"Keep-alive ping started: {health_url} every {interval_seconds}s")
while not STOP_EVENT.is_set():
try:
time.sleep(interval_seconds)
resp = session.get(health_url, timeout=timeout_seconds)
if resp.status_code == 200:
logger.info("Keep-alive OK")
except Exception as e:
logger.warning(f"Keep-alive error: {e}")

def start_keep_alive_thread(interval_seconds=300, timeout_seconds=5, url_env_var="RENDER_PUBLIC_URL"):
t = threading.Thread(target=keep_alive_ping, kwargs=dict(interval_seconds=interval_seconds, timeout_seconds=timeout_seconds, url_env_var=url_env_var), daemon=True)
t.start()
return t

def shutdown(*):
STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

#==================== Flask Routes ====================
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
bot.account_equity = max(available_cash, DEFAULT_ACCOUNT_EQUITY)
bot.max_positions = bot._max_positions_for_equity(bot.account_equity)
except Exception:
available_cash = 0

text
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
    "access_token_valid": bot.access_token is not None,
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

text
# Find open position by symbol
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
# verify
bot.kite.profile()
return jsonify({"success": True, "message": "Token updated"})
except Exception as e:
return jsonify({"success": False, "message": str(e)}), 400

@app.route("/control/<action>", methods=["GET"])
def control(action):
if not bot.access_token:
return jsonify({"status": "Authenticate first"}), 401
if action == "scan":
threading.Thread(target=bot.scan_for_opportunities, daemon=True).start()
return jsonify({"status": "scan queued"})
elif action == "pause":
bot.bot_status = "Paused"
return jsonify({"status": "paused"})
elif action == "resume":
bot.bot_status = "Running"
return jsonify({"status": "resumed"})
elif action == "rebuild_and_scan":
bot.update_daily_stock_list()
threading.Thread(target=bot.scan_for_opportunities, daemon=True).start()
return jsonify({"status": "rebuild+scan queued"})
return jsonify({"status": f"unknown action {action}"}), 400

@app.route("/backtest/run", methods=["POST"])
def backtest_run():
# Minimal placeholder: strategy backtest is non-trivial with intraday slippage.
# Returns CSV path after simulated runs (randomized outcome placeholder).
start = now_ist().date() - timedelta(days=250)
end = now_ist().date()
trades = []
equity = DEFAULT_ACCOUNT_EQUITY
# Dummy: record 20 sessions with small expectancy
for i in range(20):
pnl = np.random.normal(loc=equity0.002, scale=equity0.004) # illustrative
equity += pnl
trades.append({"day": i+1, "pnl": round2(pnl), "equity": round2(equity)})
out_csv = "backtest_pnl.csv"
pd.DataFrame(trades).to_csv(out_csv, index=False)
return jsonify({"success": True, "final_equity": round2(equity), "trades": len(trades), "csv": "/backtest/csv"})

@app.route("/backtest/csv", methods=["GET"])
def backtest_csv():
fname = "backtest_pnl.csv"
if not os.path.exists(fname):
return jsonify({"success": False, "message": "No CSV"}), 404
return send_file(fname, as_attachment=True)

#==================== App main ====================
if name == "main":
print("Intraday Trading Bot (VWAP+ATR+MTF)")
print("Order Exec: Zerodha | Data: Yahoo Finance (demo)")
print("Risk/trade 1% | MTF EMA20 + VWAP | ATR breakout & trailing")
start_keep_alive_thread(interval_seconds=int(os.environ.get("KEEPALIVE_SEC","120")))
# Rapid monitor thread
def rapid_monitor():
print("Rapid monitor thread started")
while True:
try:
if bot.positions:
bot.monitor_positions()
except Exception as e:
print(f"[Monitor Thread Error]: {e}")
time.sleep(20)
threading.Thread(target=rapid_monitor, daemon=True).start()

text
# Scheduler loop thread
def run_scheduled_tasks():
    print("Scheduler loop started")
    while True:
        try:
            if bot.access_token:
                schedule.run_pending()
        except Exception as e:
            print(f"[Scheduled Task Error]: {e}")
        time.sleep(5)
threading.Thread(target=run_scheduled_tasks, daemon=True).start()

# Flask
port = int(os.environ.get("PORT", "10000"))
app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

