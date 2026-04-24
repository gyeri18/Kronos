# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model and akshare.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_day.py --symbol 000001

Arguments:
    --symbol     Stock code (e.g. 002594 for BYD, 000001 for SSE Index)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_data.csv and ./outputs/pred_<symbol>_chart.png
    - Logs and progress are printed to console

Example:
    bash> python prediction_cn_markets_day.py --symbol 000001
    python3 prediction_cn_markets_day.py --symbol 002594

Notes (personal):
    - Increased LOOKBACK from 400 to 480 to give the model more historical context.
    - Using SAMPLE_COUNT=3 for averaging multiple samples reduces prediction variance.
    - Reduced PRED_LEN from 120 to 60; 4 months out feels too speculative for daily data.
    - Bumped max_retries from 3 to 5; akshare occasionally has transient connection issues
      during market hours and 3 retries wasn't always enough.
    - Increased retry sleep from 1.5s to 2.0s; gives akshare a bit more breathing room
      between attempts, especially when the server seems under load.
"""

import os
import argparse
import time
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# Setting
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 480   # increased from 400 for more historical context
PRED_LEN = 60    # reduced from 120; shorter horizon is more reliable for daily predictions
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 3  # increased from 1; average multiple samples to reduce variance

def load_data(symbol: str) -> pd.DataFrame:
    print(f"📥 Fetching {symbol} daily data from akshare ...")

    max_retries = 5  # increased from 3; akshare can be flaky during market hours
    df = None

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(2.0)  # increased from 1.5s; more breathing room between retries

    # If still empty after retries
    if df is None or df.empty:
        print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts. Exiting.")
        sys.exit(1)
    
    df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in nume