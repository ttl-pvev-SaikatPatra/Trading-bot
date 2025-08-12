import logging
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, time as datetime_time
import schedule
import os
import yfinance as yf
import requests
import json
import pytz
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

# Your API credentials (stored securely in Replit secrets)
API_KEY = os.environ.get('KITE_API_KEY')
API_SECRET = os.environ.get('KITE_API_SECRET')

# Validation check
if not API_KEY or not API_SECRET:
    print("❌ ERROR: API credentials not found!")
    print("Please add KITE_API_KEY and KITE_API_SECRET in Replit Secrets tab")
    exit()

print(f"✅ API Key loaded: {API_KEY[:8]}... (Personal API - FREE)")
print("✅ API Secret loaded successfully")

IST = pytz.timezone('Asia/Kolkata')
MARKET_START = datetime_time(9, 15)
MARKET_END = datetime_time(15, 30)

# Create Flask app instance at module level
app = Flask(__name__)
CORS(app, origins=['https://trading-app-phi-liart.vercel.app'])  # Enable CORS for all domains
bot_instance = None  # Will be set when bot is created

class FreeAutoTradingBot:

    def __init__(self):
        global bot_instance
        bot_instance = self  # Set global reference for Flask routes

        self.kite = KiteConnect(api_key=API_KEY)
        self.access_token = None
        self.positions = {}
        self.pending_orders = []
        self.bot_status = 'Initializing'
        self.total_trades_today = 0
        self.win_rate = 0
        self.target_profit = 1.8  # 1.8% target profit change (realistic for intraday)
        self.stop_loss = 0.6   # 0.6% stop loss (R:R = 2:1)
        self.daily_stock_list = []
        self._cache_lock = threading.Lock()
        self._last_cache_update = None

        # Trailing stop-loss parameters
        self.trailing_buffer_pct = 0.3  # trailing stop buffer = 0.3% away from current price
        self.trailing_start_pct = 0.5  # start trailing once profit >= 0.5%

        # Risk management
        self.RISK_PER_TRADE = 0.04  # 4% of capital per trade
        self.MAX_POSITIONS = 5  # Maximum 5 simultaneous positions

        # Set up logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('TradingBot')
        self.logger.info("Bot initialized and ready.")
        self.logger = logging.getLogger(__name__)

        # Enhanced watchlist: Nifty 50 + promising mid-caps

    # self.symbol_mapping = {
    # Nifty 50 stocks (core basket)
    #    'RELIANCE': 'RELIANCE.NS',
    #   'TCS': 'TCS.NS',
    #  'HDFCBANK': 'HDFCBANK.NS',
    # 'INFY': 'INFY.NS',
    # 'ICICIBANK': 'ICICIBANK.NS',
    #  'SBIN': 'SBIN.NS',
    # 'BHARTIARTL': 'BHARTIARTL.NS',
    #'ITC': 'ITC.NS',
    #'KOTAKBANK': 'KOTAKBANK.NS',
    # 'LT': 'LT.NS',
    #'WIPRO': 'WIPRO.NS',
    # 'MARUTI': 'MARUTI.NS',
    # 'ASIANPAINT': 'ASIANPAINT.NS',
    # 'AXISBANK': 'AXISBANK.NS',
    # 'HCLTECH': 'HCLTECH.NS',
    # High-momentum mid-caps (promising in recent months)
    #'BOSCHLTD': 'BOSCHLTD.NS',
    # 'POLYCAB': 'POLYCAB.NS',
    # 'INDIAMART': 'INDIAMART.NS',
    # 'TATAELXSI': 'TATAELXSI.NS',
    # 'CUMMINSIND': 'CUMMINSIND.NS',
    # 'COFORGE': 'COFORGE.NS',
    # 'NAUKRI': 'NAUKRI.NS',
    # 'DEEPAKNTR': 'DEEPAKNTR.NS',
    # 'PIDILITIND': 'PIDILITIND.NS',
    # 'TATACOMM': 'TATACOMM.NS'
    # }

    #def pick_top_stocks_by_volatility(self, volume_threshold=300000, top_n=15):
     #   stock_list = [
      #      'RELIANCE.NS', 'TCS.NS', 'ICICIBANK.NS', 'LT.NS', 'POLYCAB.NS',
       #     'HDFCBANK.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'TATAMOTORS.NS',
        #    'ITC.NS', 'BAJFINANCE.NS', 'INFY.NS', 'AXISBANK.NS', 'MARUTI.NS',
         #   'ADANIGREEN.NS', 'BHEL.NS', 'ADANIPORTS.NS', 'CANBK.NS',
          #  'HAVELLS.NS', 'COALINDIA.NS', 'TATAELXSI.NS', 'RBLBANK.NS',
           # 'BANKBARODA.NS', 'POLYCAB.NS', 'DEEPAKNTR.NS', 'PIDILITIND.NS',
            #'CUMMINSIND.NS', 'COFORGE.NS', 'NAUKRI.NS'
       # ]
        #ranking = []
       # for symbol in stock_list:
        #    try:
         #       df = yf.download(symbol,
          #                       period='10d',
           #                      interval='1d',
            #                     progress=False,
             #                    auto_adjust=False,
              #                   threads=False)

                # # Flatten MultiIndex columns to single level if needed
                # if isinstance(df.columns, pd.MultiIndex):
                #     df.columns = [col[0] for col in df.columns]

                # if df.empty:
                #     continue

                # required_cols = ['Volume', 'High', 'Low', 'Close']
                # if not all(col in df.columns for col in required_cols):
                #     missing_cols = list(set(required_cols) - set(df.columns))
                #     self.logger.warning(
                #         f"Data error for {symbol}: missing columns - {missing_cols}"
                #     )
                #     continue

                # df = df.dropna(subset=required_cols)
                # if df.empty:
                #     continue

                # avg_volume = df['Volume'].tail(5).mean()
                # self.logger.info(f"{symbol}: Recent avg volume = {avg_volume}")
                # if pd.isna(avg_volume) or avg_volume < volume_threshold:
                #     continue

        #         high = df['High']
        #         low = df['Low']
        #         close = df['Close']

        #         tr = pd.concat([
        #             high - low, (high - close.shift()).abs(),
        #             (low - close.shift()).abs()
        #         ],
        #                        axis=1).max(axis=1)

        #         atr = tr.rolling(window=5).mean().iloc[-1]
        #         if pd.isna(atr):
        #             continue

        #         ranking.append((symbol, atr, avg_volume))
        #         import time
        #         time.sleep(0.5)  # respect remote API

        #     except Exception as e:
        #         self.logger.warning(f"Data error for {symbol}: {e}")
        #         continue

        # ranking.sort(key=lambda x: x[1], reverse=True)
        # top_stocks = [x[0].replace('.NS', '') for x in ranking[:top_n]]
        # self.logger.info(f"Today's Top {top_n} Volatile Stocks: {top_stocks}")
        # return top_stocks
    
    def select_precise_stocks_for_trading(self):
    """Final precise stock selection combining all factors"""
    
    print("🎯 Starting precise stock selection...")
    
    # Step 1: Get today's market leaders
    market_leaders = self.get_todays_market_leaders()
    all_candidates = (market_leaders['top_gainers'][:8] + 
                     market_leaders['top_losers'][:8])
    
    # Step 2: Identify strong sectors
    strong_sectors = self.identify_strong_sectors_today()
    
    # Step 3: Add sector leaders to candidates
    for sector, data in list(strong_sectors.items())[:3]:  # Top 3 sectors
        for stock in data['strong_stocks'][:2]:  # Top 2 stocks per sector
            if stock['symbol'] not in [s['symbol'] for s in all_candidates]:
                all_candidates.append({
                    'symbol': stock['symbol'],
                    'price_change': stock['change'],
                    'range': abs(stock['change']) * 1.2,  # Estimate
                    'sector': sector
                })
    
    # Step 4: Apply liquidity filter
    liquid_stocks = self.filter_by_liquidity_and_volume(all_candidates)
    
    # Step 5: Final scoring and selection
    final_selection = []
    
    for stock in liquid_stocks[:15]:  # Top 15 after filtering
        symbol = stock['symbol']
        
        # Calculate composite score
        movement_score = abs(stock['price_change']) * 0.3
        range_score = stock['range'] * 0.4
        volume_score = min(stock.get('volume_ratio', 1), 3) * 0.3
        
        composite_score = movement_score + range_score + volume_score
        
        # Set dynamic targets based on actual movement
        if stock['range'] > 3.0:
            target = min(stock['range'] * 0.4, 2.0)
            stop = target * 0.5
        elif stock['range'] > 2.0:
            target = min(stock['range'] * 0.5, 1.5)
            stop = target * 0.5
        else:
            target = min(stock['range'] * 0.6, 1.0)
            stop = target * 0.5
        
        final_selection.append({
            'symbol': symbol,
            'score': composite_score,
            'target_profit': round(target, 2),
            'stop_loss': round(stop, 2),
            'movement_today': stock['price_change'],
            'range_today': stock['range'],
            'selection_reason': 'market_leader' if stock in market_leaders['top_gainers'][:5] + market_leaders['top_losers'][:5] else 'sector_strength'
        })
    
    # Sort by composite score and return top 8-10
    final_selection.sort(key=lambda x: x['score'], reverse=True)
    
    selected_stocks = final_selection[:8]
    
    print(f"✅ Selected {len(selected_stocks)} stocks for trading:")
    for stock in selected_stocks:
        print(f"   {stock['symbol']}: Target {stock['target_profit']}%, Stop {stock['stop_loss']}% | Reason: {stock['selection_reason']}")
    
    return selected_stocks


    def authenticate_with_token(self, request_token):
        """Authenticate using request token from Vercel app"""
        try:
            print(f"🔄 Authenticating with request token: {request_token[:8]}...")
            data = self.kite.generate_session(request_token, api_secret=API_SECRET)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
        
            # Save token to file
            current_time = datetime.now(IST).replace(microsecond=0)
            with open('access_token.txt', 'w') as f:
                f.write(f"{self.access_token}\n{current_time.isoformat()}")
            
            print(f"✅ Authentication successful! Token saved.")
            self.bot_status = 'Active'
        
            # Test connection
            try:
                profile = self.kite.profile()
                margins = self.kite.margins()
                balance = margins['equity']['available']['live_balance']
                print(f"👤 Welcome: {profile['user_name']}")
                print(f"💰 Available Balance: ₹{balance:,.2f}")
            except Exception as e:
                print(f"⚠️ Authentication successful but error getting account info: {e}")
            
            return True
        
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            self.bot_status = 'Authentication Failed'
            return False

    def _is_within_refresh_window(self,
                                  refresh_time: datetime.time,
                                  tolerance_minutes=5):
        """
        Check if current time is within the tolerance window around refresh_time.
        This helps to ensure the refresh happens near the specified time.
        """
        ist_now = self.get_ist_time()
        now_time = ist_now.time()
        refresh_datetime = datetime.combine(ist_now.date(), refresh_time)
        lower_bound = (refresh_datetime -
                       timedelta(minutes=tolerance_minutes)).time()
        upper_bound = (refresh_datetime +
                       timedelta(minutes=tolerance_minutes)).time()
        return lower_bound <= now_time <= upper_bound

    def update_daily_stock_list(self):
        """
        Safely update the cached stock list by picking top volatile stocks.
        """
        try:
            with self._cache_lock:
                self.logger.info(
                    "⏳ Updating daily stock list based on volatility and volume metrics..."
                )
                self.daily_stock_list = self.pick_top_stocks_by_volatility(
                    volume_threshold=300000, top_n=15)
                self._last_cache_update = self.get_ist_time()
                self.logger.info(
                    f"✅ Daily stock list updated at {self._last_cache_update.strftime('%H:%M:%S IST')}"
                )
        except Exception as e:
            self.logger.error(f"🔴 Failed to update daily stock list: {e}")

    def maybe_refresh_daily_stock_list(self):
        """
        Refresh the cached stock list if current time is near scheduled refresh times.
        This runs every scan cycle with a lock to avoid overlap.
        """
        scheduled_times = [
            datetime_time(8, 45),
            datetime_time(11, 0),
            datetime_time(13, 0)
        ]

        ist_now = self.get_ist_time()
        now_time = ist_now.time()

        with self._cache_lock:
            # If no cache yet or last update was not today, force update
            if (self._last_cache_update is None
                    or self._last_cache_update.date() != ist_now.date()):
                self.logger.info(
                    "Cache empty or outdated. Forcing stock list refresh.")
                self.update_daily_stock_list()
                return

            # Check if we are within refresh windows
            for refresh_time in scheduled_times:
                # Only refresh if we have not refreshed recently and within tolerance
                if self._last_cache_update < datetime.combine(
                        ist_now.date(), refresh_time) - timedelta(minutes=5):
                    if self._is_within_refresh_window(refresh_time):
                        self.logger.info(
                            f"Scheduled stock list refresh window at {refresh_time.strftime('%H:%M')}."
                        )
                        self.update_daily_stock_list()
                        break

    # === Time Utility Functions ===
    def get_ist_time(self):
        """Get current IST time safely"""
        try:
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            return ist_now
        except Exception as e:
            self.logger.error(f"Error getting IST time: {e}")
            return datetime.now()

    def is_market_open(self):
        """Check if Indian stock market is currently open (IST timezone)"""
        try:
            ist_now = self.get_ist_time()
            current_time = ist_now.time()
            current_day = ist_now.weekday()  # 0=Monday, 6=Sunday

            # Market closed on weekends
            if current_day >= 5:  # Saturday=5, Sunday=6
                return False

            # Market hours: 9:15 AM to 3:30 PM IST
            market_open = datetime_time(9, 15)
            market_close = datetime_time(15, 30)
            is_open = market_open <= current_time <= market_close

            # Debug log to verify
            self.logger.info(
                f"🕐 Current IST time: {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}"
            )
            self.logger.info(
                f"📊 Market status: {'OPEN' if is_open else 'CLOSED'}")
            return is_open
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False

    def authenticate(self):
        """Enhanced authentication with better error handling"""
        print("\n🔐 ZERODHA PERSONAL API AUTHENTICATION")
        print("=" * 50)
        print("🆓 Using FREE Personal API - No subscription fees!")
        print("🔗 Step 1: Visit this URL to login:")
        print(f"{self.kite.login_url()}")
        print(
            "\n📝 Step 2: After login, copy the request_token from the redirected URL"
        )
        print(
            "Example: https://your-redirect-url/?request_token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        )
        print("\n🎯 Step 3: Copy ONLY the 32-character request_token:")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            request_token = input(
                f"\nEnter request_token (Attempt {attempt}/{max_attempts}): "
            ).strip()

            if len(request_token) != 32:
                print(
                    "❌ Invalid request_token! It should be exactly 32 characters."
                )
                if attempt < max_attempts:
                    continue
                else:
                    return False

            try:
                print("🔄 Generating access token...")
                data = self.kite.generate_session(request_token,
                                                  api_secret=API_SECRET)
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)

                # Save token for future use
                with open('access_token.txt', 'w') as f:
                    f.write(
                        f"{self.access_token}\n{datetime.now().isoformat()}")
                    self.bot_status = 'Active'

                print("✅ Authentication successful!")

                # Test connection and show account info
                try:
                    profile = self.kite.profile()
                    margins = self.kite.margins()
                    balance = margins['equity']['available']['live_balance']

                    print(f"👤 Welcome: {profile['user_name']}")
                    print(f"💰 Available Balance: ₹{balance:,.2f}")
                    print(
                        f"🎯 Bot Target: {self.target_profit}% profit | Stop Loss: {self.stop_loss}%"
                    )
                    print("📊 Data Source: Yahoo Finance + NSE (Free)")
                    self.bot_status = 'Active'
                    return True

                except Exception as e:
                    print(
                        f"⚠️ Authentication successful but error getting account info: {e}"
                    )
                    self.bot_status = 'Active'
                    return True

            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ Attempt {attempt} failed: {e}")

                if "invalid" in error_msg or "expired" in error_msg:
                    print(
                        "💡 Token expired or invalid. Try getting a fresh token."
                    )
                    if attempt < max_attempts:
                        print(f"🔗 New login URL: {self.kite.login_url()}")
                        continue
                else:
                    print(
                        "💡 Check your internet connection and API credentials."
                    )

                if attempt == max_attempts:
                    return False

        return False

    def load_saved_token(self):
        try:
            with open('access_token.txt', 'r') as f:
                lines = f.read().strip().split('\n')
                token = lines[0]
                saved_time = datetime.fromisoformat(lines[1])
                saved_time = IST.localize(saved_time)

                # Personal API tokens expire at 6 AM next day
                now = datetime.now(IST)
                expiry_time = saved_time.replace(hour=6,
                                                 minute=0,
                                                 second=0,
                                                 microsecond=0)
                if saved_time.hour >= 6:
                    expiry_time += timedelta(days=1)

                if now < expiry_time:
                    self.access_token = token
                    self.kite.set_access_token(token)
                    self.bot_status = 'Active'
                    print(
                        f"✅ Using saved token (valid until {expiry_time.strftime('%Y-%m-%d 6:00 AM')})"
                    )
                    return True
                else:
                    print(
                        "⏰ Saved token expired. New authentication required.")
                    return False

        except Exception as e:
            print("🔄 No valid saved token. New authentication required.")
            return False

    # === Data Fetching ===
    def fetch_historical_data(self, symbol, interval='15m', days=5):
        """
        Fetch last `days` of OHLCV data for symbol at given interval using yfinance.
        Return pandas DataFrame with ['Open','High','Low','Close','Volume'] columns.
        """
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            period_str = f"{days}d"
            df = stock.history(period=period_str, interval=interval)
            if df.empty:
                self.logger.warning(
                    f"No historical data for {symbol} at {interval}")
                return None
            df = df.dropna(subset=['Close'])
            df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            },
                      inplace=True)
            return df
        except Exception as e:
            self.logger.error(
                f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_stock_price_external(self, symbol):
        """Get current stock price using yfinance"""
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                self.logger.info(f"📊 {symbol} price from Yahoo: ₹{price:.2f}")
                return float(price)
            else:
                return None
        except Exception as e:
            self.logger.error(
                f"Error fetching current price for {symbol}: {e}")
            return None

    # === Indicator Calculations ===
    def calculate_ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def calculate_macd(self, close_series):
        ema_short = self.calculate_ema(close_series, 12)
        ema_long = self.calculate_ema(close_series, 26)
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal_line

    def calculate_bollinger_bands(self, close_series, window=20, num_std=2):
        sma = close_series.rolling(window=window).mean()
        std = close_series.rolling(window=window).std()
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return upper_band, lower_band

    def calculate_atr(self, df, window=14):
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift())
        low_close_prev = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close_prev, low_close_prev],
                       axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low, (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ],
                       axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return adx

    def get_average_volume(self, df, window=20):
        return df['volume'].rolling(window=window).mean().iloc[-1]

    def check_liquidity(self, volume_avg):
        MIN_VOLUME_THRESHOLD = 100000  # Minimum volume to consider liquid
        return volume_avg >= MIN_VOLUME_THRESHOLD

    # === Multi-Timeframe Confirmation ===
    def multi_timeframe_confirmation(self, symbol):
        data_15m = self.fetch_historical_data(symbol, interval='15m', days=5)
        data_5m = self.fetch_historical_data(symbol, interval='5m', days=1)

        if data_15m is None or data_5m is None:
            return False

        close_15m = data_15m['close']
        macd_15m, signal_15m = self.calculate_macd(close_15m)
        rsi_15m = self.calculate_rsi(close_15m)

        close_5m = data_5m['close']
        macd_5m, signal_5m = self.calculate_macd(close_5m)
        rsi_5m = self.calculate_rsi(close_5m)

        ema_20_15m = self.calculate_ema(close_15m, 20).iloc[-1]
        ema_20_5m = self.calculate_ema(close_5m, 20).iloc[-1]

        price_15m = close_15m.iloc[-1]
        price_5m = close_5m.iloc[-1]

        macd_cross_15m = macd_15m.iloc[-1] > signal_15m.iloc[-1]
        macd_cross_5m = macd_5m.iloc[-1] > signal_5m.iloc[-1]

        rsi_confirm_15m = rsi_15m.iloc[-1] > 40
        rsi_confirm_5m = rsi_5m.iloc[-1] > 40

        price_above_ema_15m = price_15m > ema_20_15m
        price_above_ema_5m = price_5m > ema_20_5m

        if macd_cross_15m and macd_cross_5m and rsi_confirm_15m and rsi_confirm_5m and price_above_ema_15m and price_above_ema_5m:
            return True
        return False

    # === Signal Generation ===
    def generate_trade_signal(self, symbol):
        df = self.fetch_historical_data(symbol, interval='15m', days=5)
        if df is None or df.empty:
            self.logger.warning(f"No data for {symbol}")
            return None

        closes = df['close']
        volumes = df['volume']

        rsi = self.calculate_rsi(closes)
        macd_line, signal_line = self.calculate_macd(closes)
        upper_band, lower_band = self.calculate_bollinger_bands(closes)
        atr = self.calculate_atr(df)
        adx = self.calculate_adx(df)
        avg_vol = self.get_average_volume(df)

        if not self.check_liquidity(avg_vol):
            self.logger.info(
                f"Stock {symbol} filtered out due to low liquidity.")
            return None

        price = closes.iloc[-1]
        rsi_current = rsi.iloc[-1]
        macd_val = macd_line.iloc[-1]
        macd_sig = signal_line.iloc[-1]
        upper_bb = upper_band.iloc[-1]
        lower_bb = lower_band.iloc[-1]
        atr_current = atr.iloc[-1]
        adx_current = adx.iloc[-1]
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]

        buy_score = 0
        short_score = 0

        # Buy conditions
        if macd_val > macd_sig:
            buy_score += 1
        if rsi_current > 45:
            buy_score += 1
        if price > sma_20:
            buy_score += 1
        if price > upper_bb:
            buy_score += 1
        if adx_current and adx_current > 20:
            buy_score += 1

        # Short conditions (inverse logic)
        if macd_val < macd_sig:
            short_score += 1
        if rsi_current < 55:
            short_score += 1
        if price < sma_20:
            short_score += 1
        if price < lower_bb:
            short_score += 1
        if adx_current and adx_current > 20:
            short_score += 1

        # Multi timeframe confirmation (only for buy signals here to increase accuracy)
        mft_confirmed = self.multi_timeframe_confirmation(symbol)

        if buy_score >= 4 and mft_confirmed:
            return 'BUY', atr_current
        elif short_score >= 4 and mft_confirmed:
            return 'SHORT', atr_current
        else:
            return None

    # === Position Sizing ===
    def calculate_position_size(self,
                                risk_per_trade,
                                capital,
                                atr,
                                current_price,
                                stop_loss_atr_multiplier=1):
        stop_loss_distance = atr * stop_loss_atr_multiplier
        if stop_loss_distance == 0:
            return 0
        risk_amount = capital * risk_per_trade
        qty = int(risk_amount / stop_loss_distance / current_price)
        return max(qty, 0)

    # === Trailing Stop Update ===
    def trailing_stop_update(self,
                             entry_price,
                             atr,
                             current_price,
                             position_type='LONG',
                             trailing_multiplier=1.5):
        trail_distance = atr * trailing_multiplier
        if position_type == 'LONG':
            return max(entry_price, current_price - trail_distance)
        else:
            return min(entry_price, current_price + trail_distance)

    # === Place Order ===
    def place_order(self, symbol, quantity, is_buy):
        try:
            if not self.is_market_open():
                self.logger.warning("⚠️ Market is closed. Cannot place order.")
                return None

            transaction_type = self.kite.TRANSACTION_TYPE_BUY if is_buy else self.kite.TRANSACTION_TYPE_SELL

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET)

            order_type_str = "BUY" if is_buy else "SHORT"
            self.logger.info(
                f"✅ {order_type_str} ORDER PLACED: {symbol} | Qty: {quantity} | Order ID: {order_id}"
            )
            return order_id
        except Exception as e:
            self.logger.error(f"❌ Error placing order for {symbol}: {e}")
            return None

    # === Get Account Balance ===
    def get_account_balance(self):
        try:
            margins = self.kite.margins()
            return margins['equity']['available']['live_balance']
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0

    # === Execute Trade Strategy ===
    def execute_strategy(self, symbol, direction, signal_price):
        try:
            ist_now = self.get_ist_time()

            # Avoid late entries (>14:45 IST)
            if ist_now.time() >= datetime_time(14, 45):
                self.logger.info(
                    f"⏰ Skipping {symbol}: Too late for MIS entry (after 2:45 PM)"
                )
                return False

            balance = self.get_account_balance()
            if balance < 3000:
                self.logger.warning("⚠️ Insufficient balance for trading")
                return False

            # Risk management - 2% capital per trade
            risk_amount = balance * self.RISK_PER_TRADE
            stop_loss_per_share = signal_price * (self.stop_loss / 100)
            if stop_loss_per_share == 0:
                self.logger.warning(
                    "⚠️ Stop loss per share is zero, skipping trade")
                return False

            quantity = max(1, int(risk_amount / stop_loss_per_share))

            # Limit max investment to 10% of balance
            max_investment = balance * 0.10
            required_amount = signal_price * quantity

            if required_amount > max_investment:
                quantity = int(max_investment / signal_price)

            if quantity < 1:
                self.logger.warning(f"⚠️ Position size too small for {symbol}")
                return False

            # Place order
            is_buy = (direction == "BUY")
            order_id = self.place_order(symbol, quantity, is_buy)
            if not order_id:
                return False

            # Target and Stop Loss
            if direction == "BUY":
                target_price = signal_price * (1 + self.target_profit / 100)
                stop_loss_price = signal_price * (1 - self.stop_loss / 100)
            else:  # SHORT
                target_price = signal_price * (1 - self.target_profit / 100)
                stop_loss_price = signal_price * (1 + self.stop_loss / 100)

            position_key = f"{symbol}_{order_id}"
            self.positions[position_key] = {
                'symbol': symbol,
                'side': direction,
                'entry_price': signal_price,
                'quantity': quantity,
                'target_price': target_price,
                'stop_loss_price': stop_loss_price,
                'entry_time': ist_now,
                'order_id': order_id,
                'transaction_type': direction
            }

            self.logger.info(
                f"🎯 NEW {direction} POSITION: {symbol} | Entry: ₹{signal_price} | Qty: {quantity}"
            )
            self.logger.info(
                f"   🎯 Target: ₹{target_price:.2f} ({self.target_profit}%)")
            self.logger.info(
                f"   🛑 Stop Loss: ₹{stop_loss_price:.2f} ({self.stop_loss}%)")

            self.save_positions()
            return True
        except Exception as e:
            self.logger.error(f"Error executing strategy for {symbol}: {e}")
            return False

    # === Update Trailing Stops ===
    def update_trailing_stops(self):
        for key, pos in self.positions.items():
            symbol = pos['symbol']
            side = pos['side']
            entry_price = pos['entry_price']
            current_price = self.get_stock_price_external(symbol)
            if current_price is None:
                continue

            if side == "BUY":
                profit_pct = (current_price - entry_price) / entry_price * 100
                if profit_pct >= self.trailing_start_pct:
                    trailing_stop = current_price - (
                        entry_price * self.trailing_buffer_pct / 100)
                    if trailing_stop > pos['stop_loss_price']:
                        self.logger.info(
                            f"🔼 Moving trailing stop up for {symbol} from ₹{pos['stop_loss_price']:.2f} to ₹{trailing_stop:.2f}"
                        )
                        pos['stop_loss_price'] = trailing_stop

            elif side == "SHORT":
                profit_pct = (entry_price - current_price) / entry_price * 100
                if profit_pct >= self.trailing_start_pct:
                    trailing_stop = current_price + (
                        entry_price * self.trailing_buffer_pct / 100)
                    if trailing_stop < pos['stop_loss_price']:
                        self.logger.info(
                            f"🔽 Moving trailing stop down for {symbol} from ₹{pos['stop_loss_price']:.2f} to ₹{trailing_stop:.2f}"
                        )
                        pos['stop_loss_price'] = trailing_stop

        self.save_positions()

    # === Monitor Positions ===
    def monitor_positions(self):
        if not self.positions:
            return

        self.update_trailing_stops()
        self.logger.info("🔍 MONITORING ACTIVE POSITIONS...")

        positions_to_close = []
        for position_key, position in list(self.positions.items()):
            try:
                symbol = position['symbol']
                side = position['side']
                entry_price = position['entry_price']
                quantity = position['quantity']
                current_price = self.get_stock_price_external(symbol)

                if current_price is None:
                    self.logger.warning(
                        f"⚠️ Could not get current price for {symbol}")
                    continue

                # Calculate P&L
                if side == "BUY":
                    pnl_amount = (current_price - entry_price) * quantity
                    pnl_percent = (
                        (current_price - entry_price) / entry_price) * 100
                    target_hit = current_price >= position['target_price']
                    stop_hit = current_price <= position['stop_loss_price']
                else:  # SHORT
                    pnl_amount = (entry_price - current_price) * quantity
                    pnl_percent = (
                        (entry_price - current_price) / entry_price) * 100
                    target_hit = current_price <= position['target_price']
                    stop_hit = current_price >= position['stop_loss_price']

                position['current_price'] = current_price
                position['pnl'] = pnl_amount
                position['pnl_percent'] = pnl_percent

                # Calculate exit conditions
                ist_now = self.get_ist_time()
                should_exit = False
                exit_reason = ""

                if target_hit:
                    should_exit = True
                    exit_reason = f"🎉 TARGET HIT! Profit: ₹{pnl_amount:.2f} ({pnl_percent:+.2f}%)"
                elif stop_hit:
                    should_exit = True
                    exit_reason = f"🛑 STOP LOSS! Loss: ₹{pnl_amount:.2f} ({pnl_percent:+.2f}%)"
                elif ist_now.time() >= datetime_time(15, 10):
                    # Exit before 3:20 PM Zerodha auto square-off
                    should_exit = True
                    exit_reason = f"⏰ MIS SQUARE-OFF PREVENTION! P&L: ₹{pnl_amount:.2f} ({pnl_percent:+.2f}%)"
                    self.logger.info(
                        "🚨 Exiting position before Zerodha auto square-off at 3:20 PM"
                    )

                if should_exit:
                    exit_is_buy = side == "SHORT"  # If SHORT, buy to exit
                    exit_order_id = self.place_order(symbol, quantity,
                                                     exit_is_buy)
                    if exit_order_id:
                        self.logger.info(f"💼 POSITION CLOSED: {side} {symbol}")
                        self.logger.info(f"   {exit_reason}")
                        self.logger.info(
                            f"   Duration: {ist_now - position['entry_time']}")
                        positions_to_close.append(position_key)
                        self.total_trades_today += 1
                else:
                    self.logger.info(
                        f"📊 {side} {symbol}: ₹{current_price:.2f} | P&L: ₹{pnl_amount:+.2f} ({pnl_percent:+.2f}%)"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error monitoring position {position_key}: {e}")

        for key in positions_to_close:
            del self.positions[key]

        if positions_to_close:
            self.save_positions()

    def scan_for_opportunities(self):
        self.logger.info("🔍 SCAN_FOR_OPPORTUNITIES CALLED - Starting scan...")
        if not self.is_market_open():
           self.logger.info("📊 Market is closed. Skipping scan.")
           return

        if len(self.positions) >= self.MAX_POSITIONS:
            self.logger.info(
                "⚠️ Maximum positions reached. Skipping new trades.")
            return

        # Refresh the daily stock list cache if needed
        self.maybe_refresh_daily_stock_list()

        with self._cache_lock:
            watchlist = self.daily_stock_list.copy(
            )  # copy to be safe if updated during scan

        if not watchlist:
            self.logger.warning("⚠️ Watchlist empty. Skipping scan.")
            return

        self.logger.info(
            f"🔍 SCANNING FOR OPPORTUNITIES on {len(watchlist)} stocks")

        opportunities_found = 0
        for symbol in watchlist:
            try:
                # Skip if already holding the stock
                if any(pos['symbol'] == symbol
                       for pos in self.positions.values()):
                    continue

                signal_res = self.generate_trade_signal(symbol)
                if signal_res:
                    direction, atr = signal_res
                    current_price = self.get_stock_price_external(symbol)
                    if current_price is None:
                        continue

                    success = self.execute_strategy(symbol, direction,
                                                    current_price)
                    if success:
                        opportunities_found += 1
                        time.sleep(
                            3)  # throttle order placement to avoid flooding
                        break  # one trade per cycle
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")

        if opportunities_found == 0:
            self.logger.info("📉 No trading opportunities found this cycle.")

    # === Main Trading Cycle ===
    def run_trading_cycle(self):
        try:
            self.logger.info(
                "🤖 === ENHANCED TRADING CYCLE WITH TRAILING STOPS ===")
            self.bot_status = 'Running'

            if not self.is_market_open():
                self.logger.info(
                    "📊 Market is closed. Only monitoring positions.")
                if self.positions:
                    self.monitor_positions()
                self.bot_status = 'Market Closed'
                return

            # Monitor existing positions first
            if self.positions:
                self.monitor_positions()

            # Then scan and execute new trades if positions < max
            if len(self.positions) < self.MAX_POSITIONS:
                self.scan_for_opportunities()

            balance = self.get_account_balance()
            self.logger.info(
                f"💰 Balance: ₹{balance:,.2f} | Positions: {len(self.positions)}"
            )
            self.logger.info(
                "📊 Enhanced: Buy+Short | Trailing Stops | Realistic Targets")

            self.logger.info("🤖 === CYCLE COMPLETED ===\n")
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            self.bot_status = f'Error: {e}'

    # === Save/Load Positions ===
    def save_positions(self):
        try:
            data = {}
            for key, pos in self.positions.items():
                data[key] = pos.copy()
                data[key]['entry_time'] = pos['entry_time'].isoformat()
            with open('positions.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")

    def load_positions(self):
        try:
            with open('positions.json', 'r') as f:
                data = json.load(f)
                for key, pos in data.items():
                    pos['entry_time'] = datetime.fromisoformat(
                        pos['entry_time'])
                    self.positions[key] = pos
            self.logger.info(f"📂 Loaded {len(self.positions)} saved positions")
        except:
            self.logger.info("📂 Starting with no saved positions")


# === Keep-alive Thread for Replit ===
def keep_alive_ping():
    while True:
        try:
            time.sleep(300)  # 5 minutes
            replit_url = os.environ.get('REPL_URL', 'http://localhost:5000')
            response = requests.get(f"{replit_url}/health", timeout=10)
            if response.status_code == 200:
                print(
                    f"✅ Keep-alive ping successful at {datetime.now().strftime('%H:%M:%S')}"
                )
            else:
                print(f"⚠️ Keep-alive ping failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Keep-alive ping error: {e}")


# === Flask Routes ===

@app.route('/set-token', methods=['POST'])
def set_token_api():
    global bot_instance
    try:
        data = request.get_json()
        request_token = data.get('request_token')
        
        if not request_token or len(request_token) != 32:
            return {"success": False, "error": "Invalid token format"}, 400
        
        # Create bot if it doesn't exist
        if not bot_instance:
            bot_instance = FreeAutoTradingBot()
        
        success = bot_instance.authenticate_with_token(request_token)
        
        if success:
            return {
                "success": True, 
                "message": "Authentication successful! Token saved.",
                "redirect_url": "https://trading-bot-ynt2.onrender.com/initialize"
            }
        else:
            return {"success": False, "error": "Authentication failed"}, 400
            
    except Exception as e:
        return {"success": False, "error": str(e)}, 500

# ADD THIS ROUTE to your main.py
@app.route('/api/close-position', methods=['POST'])
def close_position():
    global bot_instance
    if not bot_instance:
        return jsonify({'success': False, 'message': 'Bot not initialized'})
    
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'success': False, 'message': 'Symbol required'})
        
        # Find and close the position
        position_to_close = None
        for key, pos in bot_instance.positions.items():
            if pos['symbol'] == symbol:
                position_to_close = (key, pos)
                break
        
        if not position_to_close:
            return jsonify({'success': False, 'message': 'Position not found'})
        
        key, pos = position_to_close
        
        # Place exit order
        is_buy_to_close = pos['side'] == 'SHORT'  # If SHORT, buy to close
        order_id = bot_instance.place_order(symbol, pos['quantity'], is_buy_to_close)
        
        if order_id:
            # Remove from positions
            del bot_instance.positions[key]
            bot_instance.save_positions()
            return jsonify({'success': True, 'message': f'Position {symbol} closed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to place exit order'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})



# REPLACE the home() route with:
@app.route('/')
def home():
    global bot_instance
    if not bot_instance:
        return "<h1>Bot initializing...</h1>"
    try:
        status = 'Active' if bot_instance.access_token else 'Inactive'
        positions_count = len(bot_instance.positions)
        market_status = 'Open' if bot_instance.is_market_open() else 'Closed'
        target_profit = bot_instance.target_profit
        stop_loss = bot_instance.stop_loss
        current_time = bot_instance.get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')
        return f'''
        <html><head><title>Trading Bot Dashboard</title></head><body>
        <h1>🤖 Enhanced Trading Bot Dashboard</h1>
        <p><b>Status:</b> {status}</p>
        <p><b>Positions Open:</b> {positions_count}</p>
        <p><b>Market Status:</b> {market_status}</p>
        <p><b>Target Profit:</b> {target_profit}%</p>
        <p><b>Stop Loss:</b> {stop_loss}%</p>
        <p><small>Last updated: {current_time}</small></p>
        </body></html>
        '''
    except Exception as e:
        return f"<h1>Error: {e}</h1>"


@app.route('/api/status')
def api_status():
    global bot_instance
    if not bot_instance:
        return jsonify({
            "error": "Bot not initialized",
            "access_token_valid": False
        })

    try:
        margins = bot_instance.kite.margins()
        available_cash = margins['equity']['available']['live_balance']
    except:
        available_cash = 0

    daily_pnl = sum(
        [pos.get('pnl', 0) for pos in bot_instance.positions.values()])

    positions_list = []
    for pos in bot_instance.positions.values():
        current_price = bot_instance.get_stock_price_external(
            pos['symbol']) or pos['entry_price']
        if pos['side'] == 'BUY':
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            pnl_percent = ((current_price - pos['entry_price']) /
                           pos['entry_price']) * 100
        else:
            pnl = (pos['entry_price'] - current_price) * pos['quantity']
            pnl_percent = ((pos['entry_price'] - current_price) /
                           pos['entry_price']) * 100

        positions_list.append({
            'symbol':
            pos['symbol'],
            'transaction_type':
            pos['side'],
            'buy_price':
            pos['entry_price'],
            'current_price':
            current_price,
            'quantity':
            pos['quantity'],
            'target_price':
            pos['target_price'],
            'stop_loss_price':
            pos['stop_loss_price'],
            'entry_time':
            pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'pnl':
            pnl,
            'pnl_percent':
            pnl_percent
        })

    return jsonify({
        'balance':
        available_cash,
        'positions':
        positions_list,
        'orders':
        bot_instance.pending_orders,
        'market_open':
        bot_instance.is_market_open(),
        'bot_status':
        bot_instance.bot_status,
        'target_profit':
        bot_instance.target_profit,
        'stop_loss':
        bot_instance.stop_loss,
        'daily_pnl':
        daily_pnl,
        'total_trades':
        bot_instance.total_trades_today,
        'win_rate':
        bot_instance.win_rate,
        'access_token_valid':
        bot_instance.access_token is not None,
        'risk_per_trade':
        bot_instance.RISK_PER_TRADE,
        'max_positions':
        bot_instance.MAX_POSITIONS,
        'last_update':
        bot_instance.get_ist_time().strftime('%H:%M:%S')
    })


@app.route('/api/refresh-token', methods=['POST'])
def refresh_access_token():
    global bot_instance
    data = request.json
    new_token = data.get('access_token')
    if not new_token:
        return jsonify({'success': False, 'message': 'Token required'})
    if bot_instance:
        bot_instance.kite.set_access_token(new_token)
        bot_instance.access_token = new_token
        with open('access_token.txt', 'w') as f:
            f.write(f"{new_token}\n{datetime.now().isoformat()}")
        try:
            profile = bot_instance.kite.profile()
            return jsonify({
                'success':
                True,
                'message':
                f'Token updated for {profile.get("user_name", "user")}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    return jsonify({'success': False, 'message': 'Bot not initialized'})


@app.route('/control/<action>')
def control_bot(action):
    global bot_instance
    if not bot_instance:
        return jsonify({'status': 'Bot not running'})

    if action == 'scan':
        threading.Thread(target=bot_instance.scan_for_opportunities, daemon=True).start()
        return jsonify({'status': 'Manual scan triggered (BUY+SHORT opportunities with trailing stops)'})
    elif action == 'pause':
        bot_instance.bot_status = 'Paused'
        return jsonify({'status': 'Bot paused'})
    elif action == 'resume':
        bot_instance.bot_status = 'Running'
        return jsonify({'status': 'Bot resumed'})
    return jsonify({'status': f'Action {action} executed'})


@app.route('/test')
def test():
    return "✅ Enhanced Trading Bot with Trailing Stops is working!"


@app.route('/health')
def health():
    global bot_instance
    return jsonify({
        'status':
        'healthy',
        'timestamp':
        bot_instance.get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')
        if bot_instance else 'N/A',
        'flask_working':
        True,
        'bot_active':
        bot_instance.access_token is not None if bot_instance else False,
        'positions_count':
        len(bot_instance.positions) if bot_instance else 0,
        'market_open':
        bot_instance.is_market_open() if bot_instance else False,
        'features': [
            'BUY+SHORT', 'Trailing_Stops', 'Realistic_Targets', 'Midcaps',
            'IST_Timezone', 'Rapid_Monitor'
        ]
    })


@app.route('/api/refresh-watchlist', methods=['POST'])
def api_refresh_watchlist():
    global bot_instance
    if not bot_instance:
        return jsonify({'success': False, 'message': 'Bot not initialized'})

    try:
        bot_instance.update_daily_stock_list()
        return jsonify({
            'success': True,
            'message': 'Watchlist updated successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/token-status')
def token_status():
    """Check if a valid token exists"""
    global bot_instance
    try:
        has_valid_token = bot_instance and bot_instance.access_token is not None
        return jsonify({
            "has_token": has_valid_token,
            "status": "Valid token found" if has_valid_token else "No valid token"
        })
    except Exception as e:
        return jsonify({"has_token": False, "status": f"Error: {e}"})

@app.route('/initialize')
def initialize_bot():
    global bot_instance
    try:
        print("🔄 Starting bot initialization...")
        
        # Use existing authenticated bot instance
        if not bot_instance:
            return "❌ Bot not authenticated. Please authenticate first via Vercel app", 400
        
        if not bot_instance.access_token:
            return "❌ Bot not authenticated. Please authenticate first via Vercel app", 400
        
        # Load saved positions
        bot_instance.load_positions()
        print("✅ Positions loaded")
        
        # Update watchlist with dynamic stocks
        bot_instance.update_daily_stock_list()
        print("✅ Watchlist updated with fresh stocks")
        
        # Set up trading cycle scheduling
        schedule.clear()  # Clear any existing schedules
        schedule.every(15).minutes.do(bot_instance.run_trading_cycle)
        print("✅ Trading cycles scheduled every 15 minutes")
        
        return """
        ✅ Bot initialization complete!<br>
        🔄 Trading cycles running every 15 minutes<br>
        📊 Rapid monitoring every 20 seconds<br>
        🎯 Ready for trading during market hours<br>
        <br>
        <a href="/">← Back to Dashboard</a>
        """
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return f"❌ Initialization failed: {e}<br><br><a href='/'>← Back to Dashboard</a>", 500



def start_scheduled_trading():
    while True:
        schedule.run_pending()
        time.sleep(1)


# === Main execution ===
if __name__ == "__main__":
    print("🚀 ENHANCED INTRADAY TRADING BOT WITH TRAILING STOPS")
    print("=" * 60)
    print("💳 Order Execution: Zerodha Personal API (FREE)")
    print("📊 Market Data: Yahoo Finance + NSE (FREE)")
    print("🎯 Target: 1.2% profit | Stop Loss: 0.6% (Realistic)")
    print("💰 Risk: 2% capital per trade | Max: 3 positions")
    print("🔄 Features: BUY+SHORT, Trailing Stops, Mid-caps, Rapid Monitor")
    print("=" * 60)
    print("🌐 Enhanced Web Dashboard is now running!")
    print("📱 Mobile app can connect to this dashboard")
    print("⚠️  Bot initialization deferred - authenticate via Vercel app first")
    print("=" * 60)

    # Only start monitoring thread - no heavy operations
    def rapid_monitor():
        while True:
            try:
                if bot_instance and hasattr(bot_instance, 'positions') and bot_instance.positions:
                    bot_instance.monitor_positions()
            except Exception as e:
                print(f"[Monitor Thread Error]: {e}")
            time.sleep(20)

    monitor_thread = threading.Thread(target=rapid_monitor, daemon=True)
    monitor_thread.start()
    
    # Start scheduling thread (will only run when bot_instance exists)
    def run_scheduled_tasks():
        while True:
            try:
                if bot_instance and hasattr(bot_instance, 'access_token') and bot_instance.access_token:
                    schedule.run_pending()
            except Exception as e:
                print(f"[Scheduled Task Error]: {e}")
            time.sleep(30)

    schedule_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    schedule_thread.start()

    print("✅ Background monitoring threads started")
    print("🚀 Flask server starting...")

    # Start Flask app immediately (no blocking operations)
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Flask server error: {e}")
