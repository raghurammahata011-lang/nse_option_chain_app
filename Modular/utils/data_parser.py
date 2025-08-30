# utils/data_parser.py
import pandas as pd

def parse_data(symbol, data):
    """
    Convert NSE option-chain JSON to a DataFrame for the nearest expiry.
    Columns: STRIKE, CALL_OI, CALL_CHNG_IN_OI, CALL_IV, CALL_LTP, PUT_OI, PUT_CHNG_IN_OI, PUT_IV, PUT_LTP
    """
    if not data:
        return pd.DataFrame()
    
    records = data.get("records", {}) or {}
    expiry_dates = records.get("expiryDates", []) or []
    
    if not expiry_dates:
        print(f"No expiry dates found for {symbol}")
        return pd.DataFrame()
    
    # Get the nearest expiry
    expiry = expiry_dates[0]
    rows = []
    
    for item in records.get("data", []) or []:
        if item.get("expiryDate") != expiry:
            continue
            
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        
        rows.append({
            "STRIKE": item.get("strikePrice"),
            "CALL_OI": ce.get("openInterest", 0),
            "CALL_CHNG_IN_OI": ce.get("changeinOpenInterest", 0),
            "CALL_IV": ce.get("impliedVolatility", 0),
            "CALL_LTP": ce.get("lastPrice", 0),
            "PUT_OI": pe.get("openInterest", 0),
            "PUT_CHNG_IN_OI": pe.get("changeinOpenInterest", 0),
            "PUT_IV": pe.get("impliedVolatility", 0),
            "PUT_LTP": pe.get("lastPrice", 0)
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No option data found for {symbol} for expiry {expiry}")
        return df
        
    return df.sort_values("STRIKE").reset_index(drop=True)