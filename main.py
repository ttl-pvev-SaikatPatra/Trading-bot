print("Starting stateless trading API...")
import logging
import os
import json
import time
import threading
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
FRONTEND_URL = os.environ.get("FRONTEND_URL", "").rstrip("/")

RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))
MAX_NOTIONAL_PCT = float(os.environ.get("MAX_NOTIONAL_PCT", "0.08"))
MAX_TRADES_PER_DAY = int(os.environ.get("MAX_TRADES_PER_DAY", "3"))
CONSECUTIVE_LOSS_LIMIT = int(os.environ.get("CONSECUTIVE_LOSS_LIMIT", "4"))
DAILY_LOSS_LIMIT_PCT = float(os.environ.get("DAILY_LOSS_LIMIT_PCT", "0.02"))
WEEKLY_LOSS_LIMIT_PCT = float(os.environ.get("WEEKLY_LOSS_LIMIT_PCT", "0.05"))
TARGET_UNIVERSE_SIZE = int(os.environ.get("TARGET_UNIVERSE_SIZE", "50"))

app = Flask(__name__)
CORS(app, origins=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TradingAPI")

STOP_EVENT = threading.Event()

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

# ==================== Stateless Bot Core ====================
class StatelessBot:
    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY) if API_KEY else None
        self.access_token = None
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

        self.target_profit_mult = 2.0  # target = entry +/- 2*ATR
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
                logger.error("Kite not initialized")
                return False
            sess = self.kite.generate_session(request_token, api_secret=API_SECRET)
            self.access_token = sess["access_token"]
            self.kite.set_access_token(self.access_token)
            with open("access_token.txt","w") as f:
                f.write(f"{self.access_token}\n{now_ist().isoformat()}")
            self.bot_status = "Active"
            self.refresh_account_info(force=True)
            return True
        except Exception as e:
            logger.error(f"Auth failed: {e}")
            self.bot_status = "Auth Failed"
            return False

    def load_saved_token(self) -> bool:
        try:
            with open("access_token.txt","r") as f:
                tok = f.readline().strip()
            if not tok:
                return False
            self.kite.set_access_token(tok)
            self.access_token = tok
            self.kite.profile()
            self.bot_status = "Active"
            self.refresh_account_info(force=True)
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
            m = self.kite.margins()  # margins API for dynamic equity [7][10]
            eq = m.get("equity", {})
            self.available_cash = safe_float(eq.get("available", {}).get("live_balance", 0.0))
            utilised = eq.get("utilised", {}) or {}
            debits = safe_float(utilised.get("debits", 0.0))
            self.used_margin = debits
            self.account_equity = max(1000.0, self.available_cash + self.used_margin)
            # dynamic max positions
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
            if df.empty: return None
            df = df.rename(columns=str.lower).reset_index()
            return df
        except Exception:
            return None

    def get_stock_price(self, symbol: str) -> Optional[float]:
        # try Kite quote first
        if self.access_token:
            try:
                q = self.kite.quote([f"NSE:{symbol}"])
                lp = safe_float(q[f"NSE:{symbol}"]["last_price"])
                if lp>0: return lp
            except Exception:
                pass
        # fallback Yahoo
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

    # -------- Universe (stateless, called by cron) --------
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
                    if h.empty or len(h)<3: continue
                    if h["Volume"].mean()>100000 and 50<=float(h["Close"].iloc[-1])<=5000:
                        selected.append(s)
                    if len(selected)>=TARGET_UNIVERSE_SIZE: break
                except Exception:
                    continue
                time.sleep(0.05)
            if len(selected)<10:
                selected = base[:min(20,len(base))]
            self.daily_stock_list = selected
            self.universe_version = now_ist().strftime("%Y-%m-%d %H:%M")
            # features placeholder (kept for UI compatibility)
            self.universe_features = pd.DataFrame({
                "Symbol":[f"{x}.NS" for x in selected],
                "Close":[100.0]*len(selected),
                "ATR_pct":[2.0]*len(selected),
                "Score":[1.0]*len(selected)
            })
            return True
        except Exception as e:
            logger.error(f"Universe update failed: {e}")
            return False

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
        if d30 is None or d5 is None or len(d30)<40 or len(d5)<40:
            return None
        ema20 = self.ema(d30["close"],20)
        if len(ema20)<2 or pd.isna(ema20.iloc[-2:]).any(): return None
        up = d30["close"].iloc[-1] > ema20.iloc[-1]
        down = d30["close"].iloc[-1] < ema20.iloc[-1]
        vwap = (d5["close"]*d5["volume"]).cumsum()/d5["volume"].cumsum()
        if pd.isna(vwap.iloc[-1]): return None
        above_vwap = d5["close"].iloc[-1] > vwap.iloc[-1]
        below_vwap = d5["close"].iloc[-1] < vwap.iloc[-1]
        return {"long_ok": up and above_vwap, "short_ok": down and below_vwap, "d5": d5}

    def generate_signal(self, symbol: str) -> Optional[Tuple[str,float,float]]:
        mtf = self.mtf_confirmation(symbol)
        if not mtf: return None
        d5 = mtf["d5"]
        tr = self.true_range(d5)
        if tr is None: return None
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr<=0: return None
        last = float(d5["close"].iloc[-1])
        prev_high = float(d5["high"].rolling(24).max().iloc[-2])
        prev_low  = float(d5["low"].rolling(24).min().iloc[-2])
        margin = 0.0025
        long_break = last > prev_high*(1+margin)
        short_break= last < prev_low*(1-margin)
        if mtf["long_ok"] and long_break: return ("BUY", float(atr), last)
        if mtf["short_ok"] and short_break: return ("SHORT", float(atr), last)
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
            # trailing update
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

# ==================== Routes (stateless; to be invoked by external cron) ====================
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

# ==================== OAuth helpers ====================
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
    state = request.args.get("state",""); nxt = request.args.get("next","/")
    rp = {}
    if state: rp["state"]=state
    if nxt: rp["next"]=nxt
    return redirect(_kite_login_url(API_KEY, redirect_params=rp if rp else None), code=302)

@app.route("/auth/callback", methods=["GET"])
def auth_callback():
    req_token = request.args.get("request_token")
    nxt = request.args.get("next","/"); state = request.args.get("state","")
    if not req_token or len(req_token)<10:
        return jsonify({"success": False, "message": "invalid request_token"})
    ok = bot.authenticate_with_request_token(req_token)
    if not FRONTEND_URL:
        return jsonify({"success": ok})
    qp = {"auth": "success" if ok else "fail"}
    if state: qp["state"]=state
    url = f"{FRONTEND_URL}{nxt if nxt.startswith('/') else '/'}"
    sep = "&" if "?" in url else "?"
    return redirect(url + sep + urllib.parse.urlencode(qp), code=302)

@app.route("/cron/auth-check", methods=["POST"])
def cron_auth_check():
    """
    Proactively verify Zerodha auth and warm account info.
    Returns:
      {
        "auth_required": bool,
        "bot_active": bool,
        "equity": float,
        "available_cash": float,
        "timestamp": str
      }
    """
    try:
        # Default assumption: not authenticated
        auth_ok = False

        # Validate current token by calling profile()
        try:
            if bot.access_token and bot.kite:
                bot.kite.profile()  # raises if token is invalid/expired
                auth_ok = True
        except Exception:
            auth_ok = False

        # If authenticated, try to refresh margins (non-blocking best-effort)
        if auth_ok:
            try:
                if hasattr(bot, "refresh_account_info"):
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
        # Fail "open" with auth_required true so orchestrator can alert
        return jsonify({
            "auth_required": True,
            "bot_active": False,
            "error": str(e),
            "timestamp": now_ist().isoformat()
        })


# ==================== App main ====================
def _shutdown(*args):
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

if __name__ == "__main__":
    port = int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
