

class Indicadores:
    def __init__ (self):
        pass

    def compute_RSI(stocks_close, window=14):
        diff = stocks_close.diff(1).dropna()
        gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
        loss = (-diff.where(diff <0,0)).rolling(window=window).mean()
        RS = gain / loss

        return 100 - (100 / (1 + RS))
    
    #Função para calcular Bandas de Bollinger
    def compute_Bollinger_Bands(stocks_close, window=20, nstd=2):
        rolling_mean = stocks_close.rolling(window=window).mean()
        rolling_std = stocks_close.rolling(window=window).std()
        bollinger_high = rolling_mean + (nstd * rolling_std)
        bollinger_low = rolling_mean - (nstd * rolling_std)
        return bollinger_high, bollinger_low
    
    def Media_movel(stocks_close, window):
        return stocks_close.rolling(window=window).mean()
