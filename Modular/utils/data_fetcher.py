# utils/data_fetcher.py
import time
import random
import concurrent.futures
from config import THREAD_WORKERS

def fetch_option_chain(symbol, session):
    """
    Fetch option-chain JSON for a symbol. Retries 3 times with backoff.
    Returns JSON or None.
    """
    from config import INDICES
    
    url = (f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
           if symbol in INDICES else f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}")
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # backoff with jitter
            time.sleep(random.uniform(1.0, 2.5) * (attempt + 1))
    return None

def fetch_all_option_chains(symbols, session):
    """
    Fetch option chain data for all symbols using threading
    """
    all_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(fetch_option_chain, symbol, session): symbol
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                all_data[symbol] = data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
    
    return all_data