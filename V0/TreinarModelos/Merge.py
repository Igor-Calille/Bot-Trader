import yfinance as yf
import pandas as pd
from typing import List, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

from TreinarModelos.IndicadoresMercado import Indicadores
from TreinarModelos.Test import Test
from TreinarModelos.MLModels import using_RandomForestRegressor, using_LinearRidge


#stocks = pd.read_csv('V0/AAPL_2020-01-01__2024-01-01.csv')
#print(stocks)

class BOTS:
    def __init__(self):
        pass

    def setup_and_train_model(model_type, X, y):
        if model_type == 'RandomForestRegressor_GridSearchCV':
            model = using_RandomForestRegressor.GridSearchCV_RandomForestRegressor(X, y)
        elif model_type == 'RandomForestRegressor_normal_split':
            model = using_RandomForestRegressor.normal_split_RandomForestRegressor(X, y)

        elif model_type == 'LinearRidge_GridSearchCV':
            model = using_LinearRidge.GridSearchCV_LinearRidge(X, y)
        elif model_type == 'LinearRidge_normal_split':
            model = using_LinearRidge.normal_split(X, y)

        else:
            raise ValueError('Invalid model type. Please choose a valid model type.')
        
        return model

    def testing(predicted_stocks):
        #print(stocks.tail(50))
        #print('\n')

        accuracy = Test.check_signal_accuracy(predicted_stocks)
        print(f'Acurácia do sinal: {accuracy:.2f}')
        print('\n')


        initial_value = 10000
        final_value, backtest_results = Test.backtest_signals(predicted_stocks,initial_value)
        print(f"Valor inicial da carteira: ${initial_value}")
        print(f"Valor final da carteira: ${final_value:.2f}")
        print(f"Retorno do investimentot: {(final_value - initial_value) / initial_value * 100:.2f}%")
        #print(backtest_results[['Close', 'action', 'Portfolio Value']].tail())


    def BOT_main(stocks, model_type:int, media_movel:Union[List[int], None]=None, rsi=False, bollinger=False):

        #Criação das features e targets para o treinamento do modelo e comparação dos resultados
        stocks['target'] = stocks['close'].shift(-1)
        features = ['open', 'high', 'low', 'close', 'tick_volume']


        #Calculo de Médias Moveis
        if media_movel is not None:
            if not isinstance(media_movel, list):
                raise ValueError('media_movel precisa ser uma lista')
            else:
                for i in media_movel:
                    stocks[f'SMA_{i}'] = Indicadores.Media_movel(stocks['close'], window=i)
                    features.append(f'SMA_{i}')

        #Calculo de RSI
        if rsi:
            stocks['RSI'] = Indicadores.compute_RSI(stocks['close'], window=14)
            features.append('RSI')

        #Calculo de Bollinger
        if bollinger:
            bollinger_high, bollinger_low = Indicadores.compute_Bollinger_Bands(stocks['close'], window=20, nstd=2)
            stocks['Bollinger_High'] = bollinger_high
            stocks['Bollinger_Low'] = bollinger_low
            features.append('Bollinger_High')
            features.append('Bollinger_Low')

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']


        #Treinamento do modelo de acordo com o modelo escolhido
        model = BOTS.setup_and_train_model(model_type, X, y)

        #fazer previões
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['close'], 1, -1)#1 para compra e -1 para venda

        #Ponderação dos sinais com maior peso no machine learning
        ponderacao = False
        if ponderacao:
            #Definindo a regra de contramedida usando RSI
            #Considerando RSI > 70 como sobrecomprado e RSI < 30 como sobrevendido
            stocks['contramedida'] = np.where((stocks['RSI'] > 70) | (stocks['RSI'] < 30), -1, 1)

            #Ponderação dos sinais com maior peso no machine learning
            stocks['final_signal'] = stocks['signal_ml'] * 0.60 + stocks['contramedida'] * 0.40
            stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')
        else:
            #Case padrão sem ponderação
            stocks['final_signal'] = stocks['signal_ml'] * 1.0
            stocks['action'] = np.where(stocks['signal_ml'] > 0, 'Buy', 'Sell')

        return stocks





    def BOT_ML_1(stocks):
        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume']

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']

        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Ponderação dos sinais com maior peso no machine learning
        stocks['final_signal'] = stocks['signal_ml']


        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks
    
    def BOT_media_1(stocks):
        #Usando ML(RandomForestRegressor), MEDIA MOVEL(17 e 72)

        # Adicionando médias móveis como features
        stocks['SMA_17'] = stocks['Close'].rolling(window=17).mean()
        stocks['SMA_72'] = stocks['Close'].rolling(window=72).mean()

        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'SMA_17', 'SMA_72']

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']

        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_LinearRegression.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Ponderação dos sinais com maior peso no machine learning
        stocks['final_signal'] = stocks['signal_ml']


        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks
    
    def BOT_RSI_1(stocks):
        #Usando ML(RandomForestRegressor), RSI

        stocks['RSI'] = Indicadores.compute_RSI(stocks['Close'])

        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume','RSI']

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']

        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Definindo a regra de contramedida usando RSI
        #Considerando RSI > 70 como sobrecomprado e RSI < 30 como sobrevendido
        stocks['contramedida'] = np.where((stocks['RSI'] > 70) | (stocks['RSI'] < 30), -1, 1)

        #Ponderação dos sinais com maior peso no machine learning
        #stocks['final_signal'] = stocks['signal_ml'] * 0.60 + stocks['contramedida'] * 0.40
        stocks['final_signal'] = stocks['signal_ml']


        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks
    
    def BOT_Bollinger_1(stocks):
        #Usando ML(RandomForestRegressor) e BOLLINGER

        stocks = Indicadores.compute_Bollinger_Bands(stocks)


        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'Bollinger_High', 'Bollinger_Low']

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']

        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)


        #Ponderação dos sinais com maior peso no machine learning
        stocks['final_signal'] = stocks['signal_ml']


        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks

    def BOT_ALL_1(stocks):
        #Usando ML(RandomForestRegressor), MEDIA MOVEL(17 e 72), RSI E BOLLINGER

        stocks['RSI'] = Indicadores.compute_RSI(stocks['Close'])
        stocks = Indicadores.compute_Bollinger_Bands(stocks)

        # Adicionando médias móveis como features
        stocks['SMA_17'] = stocks['Close'].rolling(window=17).mean()
        stocks['SMA_72'] = stocks['Close'].rolling(window=72).mean()

        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'SMA_17', 'SMA_72', 'RSI', 'Bollinger_High', 'Bollinger_Low']

        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']

        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.GridSearchCV_RandomForestRegressor(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Definindo a regra de contramedida usando RSI
        #Considerando RSI > 70 como sobrecomprado e RSI < 30 como sobrevendido
        stocks['contramedida'] = np.where((stocks['RSI'] > 70) | (stocks['RSI'] < 30), -1, 1)

        #Ponderação dos sinais com maior peso no machine learning
        #stocks['final_signal'] = stocks['signal_ml'] * 0.60 + stocks['contramedida'] * 0.40
        stocks['final_signal'] = stocks['signal_ml']


        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks
    
    def BOT_ALL_2(stocks):
        #Usando ML(RandomForestRegressor), MEDIA MOVEL(30 e 60), RSI E BOLLINGER


        stocks['RSI'] = Indicadores.compute_RSI(stocks['Close'])
        stocks = Indicadores.compute_Bollinger_Bands(stocks)

        # Adicionando médias móveis como features
        stocks['SMA_30'] = stocks['Close'].rolling(window=30).mean()
        stocks['SMA_60'] = stocks['Close'].rolling(window=60).mean()

        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'SMA_30', 'SMA_60', 'RSI', 'Bollinger_High', 'Bollinger_Low']


        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']


        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Definindo a regra de contramedida usando RSI
        #Considerando RSI > 70 como sobrecomprado e RSI < 30 como sobrevendido
        stocks['contramedida'] = np.where((stocks['RSI'] > 70) | (stocks['RSI'] < 30), -1, 1)

        #Ponderação dos sinais com maior peso no machine learning
        #stocks['final_signal'] = stocks['signal_ml'] * 0.90 + stocks['contramedida'] * 0.10
        stocks['final_signal'] = stocks['signal_ml']

        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks
    
    def BOT_ALL_3(stocks):
        #Usando ML(RandomForestRegressor), MEDIA MOVEL(20,50 e 200), RSI E BOLLINGER


        stocks['RSI'] = Indicadores.compute_RSI(stocks['Close'])
        stocks = Indicadores.compute_Bollinger_Bands(stocks)

        stocks['SMA_20'] = stocks['Close'].rolling(window=20).mean()
        stocks['SMA_50'] = stocks['Close'].rolling(window=50).mean()
        stocks['SMA_200'] = stocks['Close'].rolling(window=200).mean()


        # Criando features e targets
        stocks['target'] = stocks['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'Bollinger_High', 'Bollinger_Low']


        stocks.dropna(inplace=True)
        X = stocks[features]
        y = stocks['target']


        #model = using_RandomForestRegressor.normal_split(X,y)
        model = using_RandomForestRegressor.using_GridSearchCV(X,y)

        #Gerando sinais e decisão de trading
        stocks['predicted_price'] = model.predict(X)
        stocks['signal_ml'] = np.where(stocks['predicted_price'] > stocks['Close'], 1, -1)

        #Definindo a regra de contramedida usando RSI
        #Considerando RSI > 70 como sobrecomprado e RSI < 30 como sobrevendido
        stocks['contramedida'] = np.where((stocks['RSI'] > 70) | (stocks['RSI'] < 30), -1, 1)

        #Ponderação dos sinais com maior peso no machine learning
        #stocks['final_signal'] = stocks['signal_ml'] * 0.90 + stocks['contramedida'] * 0.10
        stocks['final_signal'] = stocks['signal_ml']

        stocks['action'] = np.where(stocks['final_signal'] > 0, 'Buy', 'Sell')

        return stocks





    






'''
plt.figure(figsize=(14, 7))
plt.plot(backtest_results['Portfolio Value'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value in $')
plt.legend()
plt.show()
'''





if __name__ == "__main__":
    print('main')
    #data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

    #data.to_csv('V0/AAPL_2020-01-01__2024-01-01.csv')

    stocks = pd.read_csv('V0/AAPL_2020-01-01__2024-01-01.csv')
    #print(stocks)
    predicted_stocks= BOTS.BOT_1(stocks)
    BOTS.testing(predicted_stocks)