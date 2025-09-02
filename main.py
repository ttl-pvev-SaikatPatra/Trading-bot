print("Starting stateless trading API...")
import logging
import os
import json
import time
import signal
from datetime import datetime, timedelta, time as datetime_time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import pytz
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
from kiteconnect import KiteConnect
import urllib.parse
from urllib.parse import unquote  # decode request_token safely

# ==================== Config & Globals ====================
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = datetime_time(9, 15)
MARKET_CLOSE = datetime_time(15, 30)

API_KEY = os.environ.get("KITE_API_KEY")
API_SECRET = os.environ.get("KITE_API_SECRET")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "").rstrip("/")

RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))
MAX_NOTIONAL_PCT = float(os.environ.get("MAX_NOTIONAL_PCT", "0.08"))
MAX_TRADES_PER_DAY = int(os.environ.get("MAX_TRADES_PER_DAY", "3"))
CONSECUTIVE_LOSS_LIMIT = int(os.environ.get("CONSECUTIVE_LOSS_LIMIT", "4"))
TARGET_UNIVERSE_SIZE = int(os.environ.get("TARGET_UNIVERSE_SIZE", "50"))

# Track request_tokens to enforce local single-use (defense-in-depth)
USED_TOKENS = set()

app = Flask(__name__)
# Robust CORS incl. preflight for Vercel UI [2][3]
CORS(
    app,
    origins=["*"],
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TradingAPI")

def now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def is_market_open_now() -> bool:
    t = now_ist()
    if t.weekday() >= 5:
        return False
    return ((t.hour > 9 or (t.hour == 9 and t.minute >= 15)) and
            (t.hour < 15 or (t.hour == 15 and t.minute <= 30)))

def safe_float(x, d=0.0):
    try:
        if pd.isna(x):
            return d
        return float(x)
    except Exception:
        return d

def robust_session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET","POST"])
    a = HTTPAdapter(max_retries=r)
    s.mount("http://", a)
    s.mount("https://", a)
    s.headers.update({"User-Agent":"Mozilla/5.0"})
    return s

# Normalize double slashes and trailing slash for stable routing [4]
@app.before_request
def normalize_path():
    p = request.path
    while '//' in p:
        p = p.replace('//', '/')
    if p != '/' and p.endswith('/'):
        p = p[:-1]
    if p != request.path:
        return redirect(p, code=301)

# ==================== Stateless Bot Core ====================
class StatelessBot:
    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY) if API_KEY else None
        self.access_token: Optional[str] = None

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: List[Dict[str, Any]] = []
        self.bot_status = "Initializing"

        # dynamic, refreshed via margins
        self.account_equity = 0.0
        self.available_cash = 0.0
        self.used_margin = 0.0
        self.max_positions = 2
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0

        self.target_profit_mult = 2.0
        self.trailing_start_R = 0.6
        self.trailing_atr_mult = 1.2

        self.daily_stock_list: List[str] = []
        self.universe_features = pd.DataFrame()
        self.universe_version = None

        self._session = robust_session()
        self._last_margins_refresh: Optional[datetime] = None

        logger.info("Stateless bot initialized")

    # -------- Auth / Margins (dynamic equity) --------
    def authenticate_with_request_token(self, request_token: str) -> bool:
        try:
            if not self.kite:
                logger.error("Kite not initialized (missing API_KEY)")
                return False
            if not request_token or len(request_token) < 10:
                logger.error("Invalid request_token")
                return False

            request_token = unquote(request_token)  # defensive decode [1]
            sess = self.kite.generate_session(request_token, api_secret=API_SECRET)  # exchange [1]
            access_token = sess.get("access_token")
            if not access_token:
                logger.error("No access_token in exchange response")
                return False

            self.kite.set_access_token(access_token)
            self.access_token = access_token

            # Validate immediately (catches api key mismatch or stale token) [1]
            self.kite.profile()

            # Persist token
            try:
                with open("access_token.txt","w") as f:
                    f.write(f"{access_token}\n{now_ist().isoformat()}")
            except Exception as e:
                logger.warning(f"Failed persisting access_token: {e}")

            # Warm balances (best-effort)
            try:
                self.refresh_account_info(force=True)
            except Exception:
                pass

            self.bot_status = "Active"
            return True
        except Exception as e:
            logger.error(f"Auth failed: {e}")
            self.bot_status = "Auth Failed"
            return False

    def load_saved_token(self) -> bool:
        try:
            if not self.kite:
                return False
            with open("access_token.txt","r") as f:
                tok = f.readline().strip()
            if not tok:
                return False
            self.kite.set_access_token(tok)
            self.access_token = tok
            self.kite.profile()  # validate [1]
            self.refresh_account_info(force=True)
            self.bot_status = "Active"
            return True
        except Exception:
            self.bot_status = "Auth Required"
            return False

    def refresh_account_info(self, force: bool=False) -> bool:
        try:
            if not self.access_token or not self.kite:
                return False
            if not force and self._last_margins_refresh and (now_ist()-self._last_margins_refresh).total_seconds()<240:
                return True
            m = self.kite.margins()  # /user/margins [1]
            eq = m.get("equity", {})
            self.available_cash = safe_float(eq.get("available", {}).get("live_balance", 0.0))
            utilised = eq.get("utilised", {}) or {}
            debits = safe_float(utilised.get("debits", 0.0))
            self.used_margin = debits
            self.account_equity = max(1000.0, self.available_cash + self.used_margin)
            if self.account_equity <= 15000: self.max_positions = 1
            elif self.account_equity <= 30000: self.max_positions = 2
            elif self.account_equity <= 60000: self.max_positions = 3
            elif self.account_equity <= 100000: self.max_positions = 4
            else: self.max_positions = 5
            self._last_margins_refresh = now_ist()
            return True
        except Exception as e:
            logger.warning(f"Margins refresh failed: {e}")
            return False

    # -------- Data / Pricing --------
    def fetch_bars(self, symbol: str, interval="5m", days=2) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(f"{symbol}.NS", period=f"{days}d", interval=interval, auto_adjust=False, progress=False)
            if df.empty:
                return None
            df = df.rename(columns=str.lower).reset_index()
            # Coerce numerics and drop rows with missing key fields
            for c in ("open","high","low","close","volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close","high","low","volume"])
            # If still insufficient length, bail
            return df if len(df) >= 40 else None
        except Exception:
            return None


    def get_stock_price(self, symbol: str) -> Optional[float]:
        # Try Kite quote first
        if self.access_token:
            try:
                q = self.kite.quote([f"NSE:{symbol}"])
                lp = safe_float(q[f"NSE:{symbol}"]["last_price"])
                if lp>0: return lp
            except Exception:
                pass
        # Fallback Yahoo
        try:
            t = yf.Ticker(f"{symbol}.NS")
            px = t.fast_info.get("last_price")
            if px: return float(px)
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # -------- Universe selection --------
    def update_universe(self) -> bool:
        try:
            base = [
                "RELIANCE","TCS","HDFCBANK","INFY","HINDUNILVR","ICICIBANK","KOTAKBANK","SBIN",
                "BHARTIARTL","ITC","ASIANPAINT","LT","AXISBANK","MARUTI","SUNPHARMA","TITAN",
                "ULTRACEMCO","BAJFINANCE","WIPRO","NESTLEIND","HCLTECH","POWERGRID","NTPC","ONGC",
                "TATAMOTORS","TECHM","GRASIM","BAJAJFINSV","HINDALCO","ADANIENT","ADANIPORTS",
                "COALINDIA","DIVISLAB","EICHERMOT","HEROMOTOCO","DRREDDY","BRITANNIA","APOLLOHOSP",
                "CIPLA","HDFCLIFE","SBILIFE","INDUSINDBK","M&M","JSWSTEEL","TATASTEEL","BPCL",
                "LTIM","TATACONSUM","DIXON","POLYCAB","PERSISTENT","COFORGE","MPHASIS","MARICO",
                "DABUR","PIDILITIND","GODREJCP","CHOLAFIN","MUTHOOTFIN"
            ]
            selected = []
            for s in base:
                try:
                    h = yf.Ticker(f"{s}.NS").history(period="5d", interval="1d")
                    if h.empty or len(h) < 3:
                        continue
                    h["Volume"] = pd.to_numeric(h["Volume"], errors="coerce")
                    h["Close"] = pd.to_numeric(h["Close"], errors="coerce")
                    h = h.dropna(subset=["Volume","Close"])
                    if h.empty:
                        continue
                    if h["Volume"].mean() > 100000 and 50 <= float(h["Close"].iloc[-1]) <= 5000:
                        selected.append(s)
                    if len(selected) >= TARGET_UNIVERSE_SIZE:
                        break
                except Exception:
                    continue
                time.sleep(0.05)

            # Fallback if selection too small
            if len(selected) < 10:
                selected = base[:min(20, len(base))]

            self.daily_stock_list = selected
            self.universe_version = now_ist().strftime("%Y-%m-%d %H:%M")
            # Build minimal features for UI; values will be filled by scanners later
            self.universe_features = pd.DataFrame({
                "Symbol": [f"{x}.NS" for x in selected],
                "Close": [np.nan] * len(selected),
                "ATR_pct": [np.nan] * len(selected),
                "Score": [1.0] * len(selected)
            })

            # Optional persistence to survive restarts (safe no-op on Render)
            try:
                with open("universe.json","w") as f:
                    json.dump({
                        "version": self.universe_version,
                        "symbols": self.daily_stock_list,
                        "data": self.universe_features.to_dict(orient="records")
                    }, f)
            except Exception as e:
                logger.warning(f"Persist universe failed: {e}")

            return True
        except Exception as e:
            logger.error(f"Universe update failed: {e}")
            return bool(self.daily_stock_list)



    # -------- Signals / Risk --------
    def ema(self, series: pd.Series, n: int) -> pd.Series:
        return series.ewm(span=n, adjust=False).mean()

    def true_range(self, df: pd.DataFrame) -> Optional[pd.Series]:
        try:
            if df is None or len(df)<2: return None
            h = df["high"].values
            l = df["low"].values
            c = df["close"].values
            hl = h - l
            hc = np.abs(h - np.roll(c,1)); hc=np.nan
            lc = np.abs(l - np.roll(c,1)); lc=np.nan
            tr = np.fmax(hl, np.fmax(hc, lc))
            return pd.Series(tr, index=df.index, name="tr")
        except Exception:
            return None

    def mtf_confirmation(self, symbol: str) -> Optional[Dict[str, Any]]:
        d30 = self.fetch_bars(symbol, "30m", 10)
        d5 = self.fetch_bars(symbol, "5m", 2)

        # Basic shape checks
        if d30 is None or d5 is None or len(d30) < 40 or len(d5) < 40:
            return None

        # Compute EMA on the 30m close series
        ema20 = self.ema(d30["close"], 20)

        # Make sure the last 2 ema20 values exist and are finite
        tail2 = ema20.iloc[-2:]
        if len(tail2) < 2:
            return None
        if bool(tail2.isna().any()):
            return None

        # Scalar comparisons only (avoid Series truth ambiguity)
        last_close_30 = float(d30["close"].iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        up = last_close_30 > last_ema20
        down = last_close_30 < last_ema20

        # VWAP on 5m with divide-by-zero protection
        vol = d5["volume"].astype(float)
        px = d5["close"].astype(float)
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap = (px.mul(vol)).cumsum() / cum_vol
        if pd.isna(vwap.iloc[-1]):
            return None

        last_close_5 = float(px.iloc[-1])
        last_vwap = float(vwap.iloc[-1])
        above_vwap = last_close_5 > last_vwap
        below_vwap = last_close_5 < last_vwap

        return {"long_ok": bool(up and above_vwap), "short_ok": bool(down and below_vwap), "d5": d5}


    def generate_signal(self, symbol: str) -> Optional[Tuple[str, float, float]]:
        mtf = self.mtf_confirmation(symbol)
        if not mtf:
            return None

        d5 = mtf["d5"]

        tr = self.true_range(d5)
        if tr is None or tr.isna().all():
            return None

        atr_series = tr.rolling(14, min_periods=14).mean()
        atr = atr_series.iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None

        last = float(d5["close"].iloc[-1])
        if not np.isfinite(last):
            return None

        # 24-bar extremes with min_periods to avoid NaN; use previous completed bar
        highs_24 = d5["high"].rolling(24, min_periods=24).max()
        lows_24 = d5["low"].rolling(24, min_periods=24).min()
        if pd.isna(highs_24.iloc[-2]) or pd.isna(lows_24.iloc[-2]):
            return None

        prev_high = float(highs_24.iloc[-2])
        prev_low = float(lows_24.iloc[-2])

        margin = 0.0025
        if mtf["long_ok"] and last > prev_high * (1 + margin):
            return ("BUY", float(atr), last)
        if mtf["short_ok"] and last < prev_low * (1 - margin):
            return ("SHORT", float(atr), last)
        return None


    def calc_qty(self, price: float, atr: float) -> int:
        self.refresh_account_info()
        if price<=0 or atr<=0 or self.account_equity<=0: return 0
        risk_rupees = self.account_equity * RISK_PER_TRADE
        raw = int(max(0, np.floor(risk_rupees / atr)))
        max_notional = self.account_equity * MAX_NOTIONAL_PCT
        cap_qty = int(max_notional // price)
        return max(0, min(raw, cap_qty))

    def place_order(self, symbol: str, quantity: int, is_buy: bool) -> Optional[str]:
        try:
            if not self.access_token or not self.kite: return None
            if not is_market_open_now(): return None
            if quantity<=0: return None
            tx = self.kite.TRANSACTION_TYPE_BUY if is_buy else self.kite.TRANSACTION_TYPE_SELL
            oid = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=tx,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            return oid
        except Exception as e:
            logger.error(f"Order failed {symbol}: {e}")
            return None

    # -------- Stateless task handlers (invoked by external cron) --------
    def cron_scan(self) -> Dict[str, Any]:
        if not self.access_token: return {"status":"noauth"}
        if not is_market_open_now(): return {"status":"closed"}
        self.refresh_account_info()
        if len(self.positions)>=self.max_positions: return {"status":"full"}
        if not self.daily_stock_list: self.update_universe()
        for s in list(self.daily_stock_list):
            if any(p["symbol"]==s for p in self.positions.values()): continue
            sig = self.generate_signal(s)
            if sig:
                side, atr, price = sig
                if now_ist().time()>=datetime_time(14,45): return {"status":"late"}
                qty = self.calc_qty(price, atr)
                if qty<1: continue
                is_buy = (side=="BUY")
                oid = self.place_order(s, qty, is_buy)
                if not oid: continue
                if is_buy:
                    stop = price - atr
                    tgt  = price + self.target_profit_mult*atr
                else:
                    stop = price + atr
                    tgt  = price - self.target_profit_mult*atr
                key = f"{s}_{oid}"
                self.positions[key] = {
                    "symbol": s, "side": side, "entry_price": price, "quantity": qty,
                    "target_price": round(tgt,2), "stop_loss_price": round(stop,2),
                    "entry_time": now_ist().isoformat(), "order_id": oid, "atr": atr
                }
                self._save_positions()
                return {"status":"entered", "symbol": s, "side": side, "qty": qty}
        return {"status":"no_signal"}

    def cron_monitor(self) -> Dict[str, Any]:
        if not self.access_token: return {"status":"noauth"}
        if not self.positions: return {"status":"nopos"}
        to_close = []
        for k, p in list(self.positions.items()):
            sym = p["symbol"]; side = p["side"]; qty = p["quantity"]
            cur = self.get_stock_price(sym)
            if cur is None: continue
            entry = p["entry_price"]
            if side=="BUY":
                pnl = (cur-entry)*qty
                tp = cur>=p["target_price"]; sl = cur<=p["stop_loss_price"]
            else:
                pnl = (entry-cur)*qty
                tp = cur<=p["target_price"]; sl = cur>=p["stop_loss_price"]
            atr = p.get("atr",0.0)
            if atr>0:
                if side=="BUY":
                    new_stop = max(p["stop_loss_price"], cur - self.trailing_atr_mult*atr)
                    if new_stop>p["stop_loss_price"]: p["stop_loss_price"]=round(new_stop,2)
                else:
                    new_stop = min(p["stop_loss_price"], cur + self.trailing_atr_mult*atr)
                    if new_stop<p["stop_loss_price"]: p["stop_loss_price"]=round(new_stop,2)
            force_eod = now_ist().time()>=datetime_time(15,10)
            if tp or sl or force_eod:
                exit_is_buy = (side=="SHORT")
                oid = self.place_order(sym, qty, exit_is_buy)
                if oid:
                    to_close.append((k, pnl, "TP" if tp else ("SL" if sl else "EOD")))
            else:
                self.positions[k]=p
        for k,pnl,reason in to_close:
            self.daily_pnl += pnl
            if pnl>0: self.consecutive_losses=0
            else: self.consecutive_losses+=1
            self.positions.pop(k, None)
        if to_close: self._save_positions()
        return {"status":"monitored", "closed": len(to_close), "daily_pnl": round(self.daily_pnl,2)}

    def _save_positions(self):
        try:
            with open("positions.json","w") as f:
                json.dump(self.positions, f)
        except Exception as e:
            logger.warning(f"Save pos failed: {e}")

    def load_positions(self):
        try:
            with open("positions.json","r") as f:
                self.positions = json.load(f)
        except Exception:
            self.positions = {}

# Global bot
bot = StatelessBot()
bot.load_saved_token()
bot.load_positions()

# ==================== Routes (UI + cron) ====================
@app.route("/")
def home():
    status = "Active" if bot.access_token else "Inactive"
    return f"<html><body><h3>Trading API</h3><p>Status: {status}</p><p>Universe size: {len(bot.daily_stock_list)}</p></body></html>"

@app.route("/health")
def health():
    bot.refresh_account_info()
    return jsonify({
        "status":"healthy",
        "timestamp": now_ist().isoformat(),
        "bot_active": bot.access_token is not None,
        "equity": bot.account_equity,
        "available_cash": bot.available_cash,
        "positions": len(bot.positions),
        "market_open": is_market_open_now()
    })

# UI-friendly APIs (for Vercel UI)
@app.route("/api/status", methods=["GET"])
def api_status():
    bot.refresh_account_info(force=False)
    return jsonify({
        "bot_active": bot.access_token is not None,
        "equity": getattr(bot, "account_equity", 0.0),
        "available_cash": getattr(bot, "available_cash", 0.0),
        "positions_count": len(getattr(bot, "positions", {})),
        "market_open": is_market_open_now(),
        "universe_size": len(getattr(bot, "daily_stock_list", [])),
        "timestamp": now_ist().isoformat()
    })

@app.route("/api/universe", methods=["GET"])
def api_universe():
    feats = getattr(bot, "universe_features", pd.DataFrame())
    return jsonify({
        "version": getattr(bot, "universe_version", None),
        "symbols": getattr(bot, "daily_stock_list", []),
        "data": feats.to_dict(orient="records") if isinstance(feats, pd.DataFrame) and not feats.empty else []
    })

@app.route("/auth/session", methods=["GET"])
def auth_session():
    ok = False
    try:
        if bot.access_token and bot.kite:
            bot.kite.profile()  # validate [1]
            ok = True
    except Exception:
        ok = False
    return jsonify({"authenticated": ok})

# Programmatic session exchange
@app.route("/session/exchange", methods=["POST"])
def session_exchange():
    data = request.get_json(force=True)
    token = data.get("request_token","")
    ok = bot.authenticate_with_request_token(token)
    return jsonify({"success": ok, "equity": bot.account_equity})

@app.route("/initialize", methods=["GET"])
def initialize():
    if not bot.access_token:
        return jsonify({"success": False, "message": "Authenticate first"}), 401
    bot.update_universe()
    return jsonify({"success": True, "universe_size": len(bot.daily_stock_list), "version": bot.universe_version})

# Cron-safe endpoints
@app.route("/cron/universe-update", methods=["GET","POST"])
def cron_universe_update():
    ok = bot.update_universe()
    return jsonify({"status": "success" if ok else "failed", "size": len(bot.daily_stock_list)})

@app.route("/cron/account-update", methods=["GET","POST"])
def cron_account_update():
    ok = bot.refresh_account_info(force=True)
    return jsonify({"status": "success" if ok else "failed", "equity": bot.account_equity, "max_positions": bot.max_positions})

@app.route("/cron/scan-opportunities", methods=["GET","POST"])
def cron_scan_opportunities():
    res = bot.cron_scan()
    return jsonify(res)

@app.route("/cron/monitor-positions", methods=["GET","POST"])
def cron_monitor_positions():
    res = bot.cron_monitor()
    return jsonify(res)

@app.route("/cron/eod-exit", methods=["POST"])
def cron_eod_exit():
    try:
        closed = 0
        if not bot.positions:
            return jsonify({"status": "no_positions", "closed": 0})
        for key, pos in list(bot.positions.items()):
            sym = pos["symbol"]; qty = pos["quantity"]; side = pos["side"]
            exit_is_buy = (side == "SHORT")
            oid = bot.place_order(sym, qty, exit_is_buy)
            if oid:
                bot.positions.pop(key, None)
                closed += 1
        bot._save_positions()
        return jsonify({"status": "eod_executed", "closed": closed})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200

@app.route("/cron/auth-check", methods=["POST"])
def cron_auth_check():
    try:
        auth_ok = False
        try:
            if bot.access_token and bot.kite:
                bot.kite.profile()  # validate [1]
                auth_ok = True
        except Exception:
            auth_ok = False
        if auth_ok:
            try:
                bot.refresh_account_info(force=True)
            except Exception:
                pass
        return jsonify({
            "auth_required": not auth_ok,
            "bot_active": auth_ok,
            "equity": getattr(bot, "account_equity", 0.0),
            "available_cash": getattr(bot, "available_cash", 0.0),
            "timestamp": now_ist().isoformat()
        })
    except Exception as e:
        return jsonify({"auth_required": True, "bot_active": False, "error": str(e), "timestamp": now_ist().isoformat()})

# ==================== OAuth helpers ====================
def _kite_login_url(api_key: str, redirect_params: dict | None = None):
    base = "https://kite.zerodha.com/connect/login?v=3"
    qp = {"api_key": api_key}
    if redirect_params:
        qp["redirect_params"] = urllib.parse.quote_plus(urllib.parse.urlencode(redirect_params))
    return f"{base}&{urllib.parse.urlencode(qp)}"  # Zerodha Connect login URL [1]

def _auth_fail_redirect(nxt, state, code):
    if not FRONTEND_URL:
        return jsonify({"success": False, "code": code}), 400
    qp = {"auth":"fail","code":code}
    if state: qp["state"]=state
    url = f"{FRONTEND_URL}{nxt if nxt.startswith('/') else '/'}"
    sep = "&" if "?" in url else "?"
    return redirect(url + sep + urllib.parse.urlencode(qp), code=302)

@app.route("/auth/login", methods=["GET"])
def auth_login():
    if not API_KEY or not API_SECRET:
        return jsonify({"success": False, "message": "Server missing API creds"}), 500
    state = request.args.get("state",""); nxt = request.args.get("next","/")
    rp = {}
    if state: rp["state"]=state
    if nxt: rp["next"]=nxt
    return redirect(_kite_login_url(API_KEY, redirect_params=rp if rp else None), code=302)

@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    req_token = request.args.get("request_token")
    nxt = request.args.get("next","/")
    state = request.args.get("state","")
    if not req_token or len(req_token) < 10:
        return jsonify({"success": False, "message": "invalid_request_token"}), 400
    try:
        global USED_TOKENS
        req_token = unquote(req_token)

        # Enforce local single-use to prevent accidental reuse while debugging/logins [5]
        if req_token in USED_TOKENS:
            logger.error("Rejected reused request_token")
            return _auth_fail_redirect(nxt, state, "token_reused")
        USED_TOKENS.add(req_token)

        logger.info(f"Exchanging token; API key starts: {str(API_KEY)[:6]}***")
        sess = bot.kite.generate_session(req_token, api_secret=API_SECRET)  # exchange [1]
        access_token = sess.get("access_token")
        if not access_token:
            return _auth_fail_redirect(nxt, state, "no_access_token")

        bot.kite.set_access_token(access_token)
        bot.access_token = access_token

        # Validate immediately; fail deterministically on error [1]
        try:
            prof = bot.kite.profile()
            logger.info(f"profile user_id: {prof.get('user_id','?')}")
        except Exception as e:
            logger.error(f"profile() failed: {e}")
            bot.access_token = None
            return _auth_fail_redirect(nxt, state, "profile_failed")

        # Persist token and warm balances
        try:
            with open("access_token.txt","w") as f:
                f.write(f"{access_token}\n{now_ist().isoformat()}")
        except Exception as e:
            logger.warning(f"Failed to persist token: {e}")
        try:
            bot.refresh_account_info(force=True)
        except Exception as e:
            logger.warning(f"Post-auth margins warm failed: {e}")

        if not FRONTEND_URL:
            return jsonify({"success": True})
        qp = {"auth":"success"}
        if state: qp["state"]=state
        url = f"{FRONTEND_URL}{nxt if nxt.startswith('/') else '/'}"
        sep = "&" if "?" in url else "?"
        return redirect(url + sep + urllib.parse.urlencode(qp), code=302)

    except Exception as e:
        logger.error(f"callback exchange error: {e}")
        return _auth_fail_redirect(nxt, state, "exchange_error")

# Optional: flush server token if FE stuck
@app.route("/auth/flush", methods=["POST"])
def auth_flush():
    try:
        bot.access_token = None
        try:
            os.remove("access_token.txt")
        except Exception:
            pass
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 200

# ==================== App main ====================
def _shutdown(*args):
    pass

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

if __name__ == "__main__":
    port = int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
