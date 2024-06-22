import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from TreinarModelos.Merge import BOTS
from TreinarModelos.Test import Test
from TreinarModelos.View import View

#Inicializar o MetaTrader 5
if not mt5.initialize():
    print('falha na inicialização, erro=', mt5.last_error())
    quit()


#Variáveis de operação
symbol = "AMAZON"
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



predicted_stocks = BOTS.BOT_main(rates_df, 'RandomForestRegressor_GridSearchCV', media_movel=[17,72], rsi=True, bollinger=True)
#BOTS.testing(predicted_stocks)


#test = [predicted_stocks,predicted_stocks]

#View.graficos(stocks=test)








accuracy = Test.check_signal_accuracy(predicted_stocks)
print(f'Acurácia do sinal: {accuracy:.2f}')
print('\n')


initial_value = 10000
final_value, backtest_results = Test.backtest_signals(predicted_stocks,initial_value)
print(f"Valor inicial da carteira: ${initial_value}")
print(f"Valor final da carteira: ${final_value:.2f}")
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
