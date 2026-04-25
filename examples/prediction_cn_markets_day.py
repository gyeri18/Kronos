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
    - Set TOP_P from 0.9 to 0.85; slightly tighter nucleus sampling felt more stable
      in my tests on volatile small-cap stocks.
    - Added a print statement after successful data fetch to show the date range loaded;
      helpful for quickly confirming how much history was actually retrieved.
    - Set DEVICE default check: if CUDA is available use it automatically, fall back to cpu.
      Saves me from manually editing this line when switching between machines.
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

# Auto-detect CUDA so I don't have to manually toggle this when switching machines
try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

MAX_CONTEXT = 512
LOOKBACK = 480   # increased from 400 for more historical context
PRED_LEN = 60    # reduced from 120; shorter horizon is more reliable for daily predictions
T = 1.0
TOP_P = 0.85     # tightened from 0.9; more stable outputs on volatile small-caps
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

    # If still empty aft
