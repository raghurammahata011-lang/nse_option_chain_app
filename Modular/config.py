# config.py
import os

# Constants and configuration
SAVE_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "NSE_STOCK")
os.makedirs(SAVE_FOLDER, exist_ok=True)
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
THREAD_WORKERS = 6