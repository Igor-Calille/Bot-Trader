import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from TreinarModelos.Merge import BOTS
from TreinarModelos.Test import Test
from TreinarModelos.View import View


# Função para obter os dados do MetaTrader 5~
def get_data_MT5(symbol = "APPLE"):
    #Inicializar o MetaTrader 5
    if not mt5.initialize():
        print('falha na inicialização, erro=', mt5.last_error())
        quit()

    #Variáveis de operação
    
    timezone = pytz.timezone("Etc/UTC")# MetaTrader 5 server timezone
    from_date = datetime(2020,1,1,tzinfo=timezone)
    to_date = datetime.now(pytz.utc)

    #Obter as mpetricas dos stocks
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, from_date, to_date)

    if rates is None:
        print('Falha no acesso do history data. Erro=',mt5.last_error())
    else:
        print(f'Dados obtidos, total de linhas: {len(rates)}')

    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    rates_df = rates_df.rename(columns={'time': 'date'})

    return rates_df

def get_data_yfinance(symbol="AAPL"):
    timezone = pytz.timezone("Etc/UTC")
    from_date = "2015-01-01"
    to_date = 2023-12-31

    # Obter stocks
    data = yf.download(symbol, start=from_date)

    if data.empty:
        print('Falha no acesso dos dados históricos.')
    else:
        print(f'Dados obtidos, total de linhas: {len(data)}')

    # Processar os dados
    data.reset_index(inplace=True)
    data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})
    #data = data.rename(columns={'Datetime': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})


    return data

def get_data_yfinance_H(symbol="BTC-USD"):
    data_H = yf.download(symbol, interval="1h")

    """
    data_H = data.resample('2H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    """
    data_H = data_H.dropna()

    data_H.reset_index(inplace=True)
    data_H = data_H.rename(columns={'Datetime': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})

    return data_H

rates_df = get_data_yfinance('BTC-USD')


print(rates_df.head())


#17,72
#Média Móvel Simples (SMA) 5,10,20
#Média Móvel Exponencial (EMA) 5,9,12
#Média Móvel Ponderada (WMA) 5, 10
'''
Média Móvel Simples (media_movel=None):

Curto prazo: 9, 10, 14 dias
Médio prazo: 20, 50 dias
Longo prazo: 100, 200 dias

Média Móvel Exponencial (media_movel_exponencial=None):

Curto prazo: 12, 20 dias
Médio prazo: 26, 50 dias
Longo prazo: 100, 200 dias

Média Móvel Ponderada (media_movel_ponderada=None):

Curto prazo: 10, 20 dias
Médio prazo: 50 dias
Longo prazo: 100, 200 dias

Média Móvel de Kaufman (media_movel_kaufman=None):

Usada menos frequentemente, mas períodos comuns são 10, 14, 30 dias, dependendo da sensibilidade desejada.

Média Móvel de Hull (media_movel_hull=None):

Frequentemente usada com períodos curtos devido à sua suavidade, como 9, 14, 20 dias.

Média Móvel Triangular (media_movel_triangular=None):

É menos comum, mas quando usada, pode variar entre 20, 50, 100 dias, dependendo do contexto.
'''


predicted_stocks = BOTS.BOT_main(
    rates_df,
    'RandomForestRegressor_GridSearchCV', 
    media_movel=[9,10,14], 
    media_movel_exponencial=[12,20], 
    media_movel_ponderada=[10,20],
    media_movel_kaufman=None,
    media_movel_hull=None,
    media_movel_triangular=None,
    rsi=True, 
    bollinger=True, 
    lags=False,
    MACD=True
)



accuracy = Test.check_signal_accuracy(predicted_stocks)
print(f'Acurácia do sinal: {accuracy:.2f}')
print('\n')


import backtrader as bt
class PandasData(bt.feeds.PandasData):
    lines = ('signal_ml',)
    params = (('signal_ml', -3),)

data_feed = PandasData(dataname=predicted_stocks, datetime='date', open='open', high='high', low='low', close='close', volume='volume')

class MLStrategy_two(bt.Strategy):
    params = (
        ('start_date', datetime(2023, 8, 30)),  # Data para começar a estratégia
        ('risk_per_trade', 1.0),
    )

    def __init__(self):
        self.start_trading = False
        self.buy_price = None
        self.sell_price = None
        self.current_value = self.broker.get_cash()

    def next(self):
        current_date = self.data.datetime.date(0)
        close_price = self.data.close[0]
        
        # Aplicar slippage
        self.buy_price = close_price * 1.001
        self.sell_price = close_price * 0.999

        # Verificar se a data atual é igual ou maior que a data de início
        if current_date >= self.params.start_date.date():
            self.start_trading = True

        if self.start_trading:
            if self.data.signal_ml[0] == 1 and self.broker.get_cash() > 0:
                amount_to_risk = self.broker.get_cash() * self.params.risk_per_trade
                size = amount_to_risk / self.buy_price
                self.buy(size=size)
                
            elif self.data.signal_ml[0] == -1 and self.position:
                self.sell(size=self.position.size)
        
        # Atualiza o valor atual do portfólio
        self.current_value = self.broker.get_value()

class MLStrategy_one(bt.Strategy):
    params = (
        ('start_date', datetime(2023, 8, 27)),  # Data para começar a estratégia
    )

    def __init__(self):
        self.start_trading = False  # Controle para iniciar a estratégia

    def next(self):
        current_date = self.data.datetime.date(0)

        if current_date >= self.params.start_date.date():
            self.start_trading = True

        if self.start_trading:
            signal_ml = self.data.signal_ml[0]  # Obtendo o sinal da coluna 'action'
            if signal_ml == 1 and not self.position:
                self.buy()  # Compra se o sinal for 'Buy'
            elif signal_ml == -1 and self.position:
                self.sell()  # Vende se o sinal for 'Sell'


class RiskSizer(bt.Sizer):
    params = (
        ('risk_per_trade', 1.0),  # Percentual do capital a arriscar em cada trade
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:  # Para compras
            amount_to_risk = cash * self.params.risk_per_trade
            size = amount_to_risk / data.close[0]  # Divide pelo preço do ativo para determinar a quantidade
            return size
        else:  # Para vendas, vende tudo que tem
            return self.broker.getposition(data).size


# Configurando o ambiente de backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy_two)

# Adicionando os dados ao cerebro
cerebro.adddata(data_feed)

# Definindo o capital inicial
cerebro.broker.set_cash(10000)

# Definindo a comissão (não usada diretamente aqui, mas pode ser adicionada)
# Configurando a execução no mesmo candle e slippage


# Executando o backtest
print(f'MLStrategy_two: Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'MLStrategy_two: Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
print(f'Rentabilidade = {round(((cerebro.broker.getvalue() - 10000) / 10000) * 100, 2)} %',  )

# Exibindo o gráfico
cerebro.plot()






# Configurando o ambiente de backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy_one)
cerebro.addsizer(RiskSizer, risk_per_trade=1.0)

# Adicionando os dados ao cerebro
cerebro.adddata(data_feed)

# Definindo o capital inicial
cerebro.broker.set_cash(10000)

# Definindo a comissão (não usada diretamente aqui, mas pode ser adicionada)
# Configurando a execução no mesmo candle e slippage
cerebro.broker.set_coc(True)
cerebro.broker.set_slippage_perc(0.01)

# Executando o backtest
print(f'MLStrategy_one: Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'MLStrategy_one: Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

# Exibindo o gráfico
cerebro.plot()





























'''
initial_value = 10000
count_trades=999
#YYYY-MM-DD
#final_value, backtest_results, count_trades = Test.backtest_signals_date(predicted_stocks,'2023-07-23', initial_value)
final_value, backtest_results, count_trades = Test.backtest_signals_date(predicted_stocks,'2023-08-26', initial_value)
#final_value, backtest_results = Test.backtest_signals(predicted_stocks, initial_value)



print(f"Valor inicial da carteira: ${initial_value}")
print(f"Valor final da carteira: ${final_value:.2f}")
print(f"Quantidade de trades: {count_trades}")
print(f"Retorno do investimentot: {(final_value - initial_value) / initial_value * 100:.2f}%")

# Plotando gráficos
plt.style.use("ggplot")
fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 linhas, 1 coluna

# Gráfico de comparação
axs[0].plot(predicted_stocks.index, predicted_stocks['close'], label='Valor Real (Close)')
axs[0].plot(predicted_stocks.index, predicted_stocks['predicted_price'], label='Valor Predito', linestyle='--')
axs[0].set_title('Comparação entre Valores Reais e Preditos')
axs[0].set_xlabel('Data')
axs[0].set_ylabel('Valor')
axs[0].legend()
axs[0].text(1, 0.35, "Teste")

plt.tight_layout()
plt.show()


'''





'''
if decision == 'buy':
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 1.0,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(symbol).ask,
        "sl": 0,
        "tp": 0,
        "deviation": 10,
        "magic": 234000,
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("Buy order has been sent: ", result)
elif decision == 'sell':
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 1.0,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(symbol).bid,
        "sl": 0,
        "tp": 0,
        "deviation": 10,
        "magic": 234000,
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print("Sell order has been sent: ", result)

    '''

mt5.shutdown()



