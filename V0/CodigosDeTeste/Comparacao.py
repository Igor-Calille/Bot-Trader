import MetaTrader5 as mt5
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

def yfinance():
    timezone = pytz.timezone("Etc/UTC")
    start_date = datetime(2024,2,2,tzinfo=timezone)
    end_date= datetime.now(timezone)
    data = yf.download('AAPL', start=start_date, end=end_date)

    print(data)


def MT5():
    if not mt5.initialize():
        quit()

    timezone = pytz.timezone("Etc/UTC")# MetaTrader 5 server timezone
    from_date = datetime(2024,2,2,tzinfo=timezone)
    to_date = datetime.now(pytz.utc)


    #Obter as mpetricas dos stocks
    rates = mt5.copy_rates_range("APPLE", mt5.TIMEFRAME_D1, from_date, to_date)

    rates = pd.DataFrame(rates)
    rates['time'] = pd.to_datetime(rates['time'], unit='s')
    print(rates)

yfinance()

MT5()