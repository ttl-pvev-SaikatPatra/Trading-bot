# File: README.md

# Intraday Trading Bot Backend

This is the backend for an automated intraday trading bot for Indian equities, built with FastAPI and designed for deployment on Render's free tier.

## Features

-   **Zerodha Integration**: Secure, redirect-based authentication.
-   **Dynamic Universe**: Daily universe creation.
-   **Strategy Engine**: EMA/VWAP crossover with RSI filter.
-   **Risk Management**: ATR-based stop-loss, position sizing, daily loss limit.
-   **Automated Execution**: Order placement, SL management, and EOD square-off.
-   **API Driven**: Control and monitor the bot via a REST API.
-   **Free Stack**: Designed to run entirely on free services (Render, GitHub Actions).

---

## 🛠️ Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd trading-bot-backend
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    -   Copy `.env.example` to a new file named `.env`.
    -   Fill in your Zerodha API credentials (`API_KEY`, `API_SECRET`).
    -   For local development, set `ZERODHA_REDIRECT_URL` to `http://127.0.0.1:8000/auth/callback`.
    -   Generate a strong `APP_SECRET_KEY` and `CRON_SECRET_KEY`.
    -   Adjust risk parameters as needed.

5.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

---

## 🚀 Deployment to Render (Free Tier)

1.  **Push to GitHub**: Create a new repository on GitHub and push this code.

2.  **Create a New Web Service on Render**:
    -   Connect your GitHub account to Render.
    -   Select your repository.
    -   Render will auto-detect the environment.
        -   **Name**: `trading-bot-backend` (or your choice)
        -   **Root Directory**: (leave blank if code is at the root)
        -   **Environment**: `Python 3`
        -   **Region**: Singapore (for lower latency to NSE servers)
        -   **Branch**: `main`
        -   **Build Command**: `pip install -r requirements.txt`
        -   **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3.  **Select the Free Instance Type**.

4.  **Add Environment Variables**:
    -   Go to the 'Environment' tab for your new service.
    -   Add all the key-value pairs from your local `.env` file.
    -   **CRITICAL**: Update `ZERODHA_REDIRECT_URL` to your Render app's URL, e.g., `https://trading-bot-backend.onrender.com/auth/callback`. Make sure this exact URL is also updated in your Zerodha Developer Console.

5.  **Deploy**: Click 'Create Web Service'. The first build will take a few minutes.

6.  **Initial Authentication**: Once deployed, you must manually go to `https://<your-app>.onrender.com/auth/login` in your browser one time to perform the Zerodha login and authorize the app. The session token will be stored securely in the SQLite database on Render's persistent disk.

---

## ⚙️ Setup GitHub Actions for Cron Jobs

To keep the Render service awake and trigger scheduled tasks, use GitHub Actions.

1.  **Add Repository Secrets**:
    -   In your GitHub repo, go to `Settings` > `Secrets and variables` > `Actions`.
    -   Create two new repository secrets:
        -   `RENDER_APP_URL`: Your full Render app URL (e.g., `https://trading-bot-backend.onrender.com`).
        -   `CRON_SECRET_KEY`: The **exact same** secret key you set in your Render environment variables.

2.  **Create Workflow File**:
    -   Create a directory `.github/workflows/` in your repository.
    -   Inside it, create a file named `scheduler.yml` with the content provided in the next section (GitHub Actions).
    -   Commit and push this file. The actions will start running on the defined schedule.
