import numpy as np

class Indicadores:
    def __init__ (self):
        pass

    def compute_RSI(stocks_close, window=14):
        diff = stocks_close.diff(1).dropna()
        gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
        loss = (-diff.where(diff <0,0)).rolling(window=window).mean()
        RS = gain / loss

        return 100 - (100 / (1 + RS))
    
    def compute_Bollinger_Bands(stocks_close, window=20, nstd=2):
        rolling_mean = stocks_close.rolling(window=window).mean()
        rolling_std = stocks_close.rolling(window=window).std()
        bollinger_high = rolling_mean + (nstd * rolling_std)
        bollinger_low = rolling_mean - (nstd * rolling_std)
        return bollinger_high, bollinger_low
    
    def Media_movel(stocks_close, window):
        return stocks_close.rolling(window=window).mean()
    
    def media_movel_exponecial(stocks_close, window):
        return stocks_close.ewm(span=window, adjust=False).mean()
    
    def media_movel_ponderada(stocks_close, window):
        weights = np.arange(1, window + 1)
        return stocks_close.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    def media_movel_kaufman(stocks_close, window):
        change = abs(stocks_close.diff(window))
        volatility = stocks_close.diff().abs().rolling(window=window).sum()
        er = change / volatility
        sc = (er * (2/(2+1) - 2/(window+1)) + 2/(window+1))**2
        kama = np.zeros_like(stocks_close)
        for i in range(window, len(stocks_close)):
            kama[i] = kama[i-1] + sc[i] * (stocks_close[i] - kama[i-1])
        return kama
    
    def media_movel_hull(self, stocks_close, window):
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        wma_half = self.media_movel_ponderada(stocks_close, half_window)
        wma_full = self.media_movel_ponderada(stocks_close, window)
        hull_ma = self.media_movel_ponderada(2 * wma_half - wma_full, sqrt_window)
        return hull_ma

    def media_movel_triangular(self, stocks_close, window):
        return self.Media_movel(self.Media_movel(stocks_close, window), window)


