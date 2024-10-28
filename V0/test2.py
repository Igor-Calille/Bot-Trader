import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from TreinarModelos.IndicadoresMercado import Indicadores
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as mtick

def GridSearchCV_RandomForestRegressor(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(random_state=42)

    param_search = {
        'n_estimators': [100, 150],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'bootstrap': [True, False]
    }

    gsearch = GridSearchCV(
        estimator=model, 
        param_grid=param_search, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        verbose=2
    )
    
    gsearch.fit(X_scaled, y)
    best_model = gsearch.best_estimator_
    
    print("Melhores hiperparâmetros para Random Forest:", gsearch.best_params_)
    
    return best_model, scaler

def GridSearchCV_LinearRegression(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = LinearRegression()

    param_search = {
        'fit_intercept': [True, False],
    }

    gsearch = GridSearchCV(estimator=model, param_grid=param_search, cv=tscv, scoring='neg_mean_squared_error')
    
    gsearch.fit(X_scaled, y)
    best_model = gsearch.best_estimator_
    
    print("Melhores hiperparâmetros para Linear Regression:", gsearch.best_params_)
    
    return best_model, scaler

def GridSearchCV_XGBoost(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_search = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    gsearch = GridSearchCV(
        estimator=model, 
        param_grid=param_search, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        verbose=2
    )
    
    gsearch.fit(X_scaled, y)
    best_model = gsearch.best_estimator_
    
    print("Melhores hiperparâmetros para XGBoost:", gsearch.best_params_)
    
    return best_model, scaler


def evaluate_model(model, X, y, scaler, tscv):
    y = np.array(y)
    X_scaled = scaler.transform(X)

    mse_scores, mae_scores, r2_scores = [], [], []
    train_mse_scores, train_mae_scores, train_r2_scores = [], [], []

    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

        train_mse_scores.append(mean_squared_error(y_train, y_train_pred))
        train_mae_scores.append(mean_absolute_error(y_train, y_train_pred))
        train_r2_scores.append(r2_score(y_train, y_train_pred))

    results = {
        'test_avg_mse': np.mean(mse_scores),
        'test_avg_mae': np.mean(mae_scores),
        'test_avg_r2': np.mean(r2_scores),
        'train_avg_mse': np.mean(train_mse_scores),
        'train_avg_mae': np.mean(train_mae_scores),
        'train_avg_r2': np.mean(train_r2_scores)
    }
    
    return results

def backtest_strategy(data, initial_balance=10000, stop_loss=0.15, min_profit_percent=0.01):
    balance = initial_balance
    initial_trade_date = "2021-01-02"
    initial_trade_date = pd.to_datetime(initial_trade_date)
    shares = 0
    balance_history = []
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        total_balance = balance + (shares * price)
        balance_history.append(total_balance)
        
        if data['date'].iloc[i] >= initial_trade_date:
            signal = data['final_signal'].iloc[i]

            if signal == 1 and balance > 0:
                shares = balance / price
                balance = 0
            elif signal == -1 and shares > 0:
                balance = shares * price
                shares = 0

            # Verifica stop loss
            if shares > 0 and (price < (balance / shares) * (1 - stop_loss)):
                balance = shares * price
                shares = 0

    final_balance = balance + (shares * data['close'].iloc[-1])  
    return_percentage = ((final_balance - initial_balance) / initial_balance) * 100

    results = {
        "final_balance": final_balance,
        "return_percentage": return_percentage,
        "balance_history": balance_history
    }
    
    return results

def check_signal_accuracy(data):
    data['price_change'] = data['close'].shift(-1) - data['close']
    conditions = [
        (data['final_signal'] > 0) & (data['price_change'] > 0),  
        (data['final_signal'] < 0) & (data['price_change'] < 0),  
    ]
    choices = [1, 1]
    data['correct_signal'] = np.select(conditions, choices, default=0)
    correct_signals = data['correct_signal'].sum()
    total_signals = np.count_nonzero(data['final_signal'])
    accuracy = correct_signals / total_signals if total_signals > 0 else 0
    return accuracy

def get_data_yfinance(symbol="AAPL"):
    from_date = "2015-01-02"
    to_date = "2023-12-28"
    data = yf.download(symbol, start=from_date, end=to_date)
    if data.empty:
        print('Falha no acesso dos dados históricos.')
    else:
        print(f'Dados obtidos, total de linhas: {len(data)}')
    data.reset_index(inplace=True)
    data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})
    return data

# Gráfico de comparação do valor do ativo com o valor de trade de cada modelo
def plot_trade_comparison(data, rf_balance_history, lr_balance_history, xgb_balance_history, initial_date, max_date, initial_investment=10000):
    # Filtrar os dados entre a data inicial e a data máxima
    data_filtered = data[(data['date'] >= initial_date) & (data['date'] <= max_date)].copy()
    rf_balance_history = rf_balance_history[-len(data_filtered):]
    lr_balance_history = lr_balance_history[-len(data_filtered):]
    xgb_balance_history = xgb_balance_history[-len(data_filtered):]

    plt.figure(figsize=(16, 9))

    # Cálculo do valor do Buy and Hold a partir do investimento inicial
    initial_price = data_filtered['close'].iloc[0]
    data_filtered['buy_and_hold'] = (data_filtered['close'] / initial_price) * initial_investment

    # Plotando o valor do Buy and Hold ao longo do tempo
    plt.plot(data_filtered['date'], data_filtered['buy_and_hold'], label='Buy and Hold', color='black', linewidth=2, linestyle='-')

    # Adicionando o valor de trade de cada modelo ao gráfico com estilos e cores distintas
    plt.plot(data_filtered['date'], rf_balance_history, label='Random Forest', linestyle='--', color='royalblue', linewidth=1.5)
    plt.plot(data_filtered['date'], lr_balance_history, label='Linear Regression', linestyle='-.', color='orange', linewidth=1.5)
    plt.plot(data_filtered['date'], xgb_balance_history, label='XGBoost', linestyle=':', color='green', linewidth=1.5)


    # Configurações adicionais do gráfico
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Valor do Investimento (R$)', fontsize=12)
    plt.title(f'Comparação do Buy and Hold com o Valor de Trade de Cada Modelo ({initial_date.date()} a {max_date.date()})', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)

    # Formatação do eixo Y para exibir valores em moeda
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('R$ {x:,.0f}'))

    # Configuração do formato de data no eixo X
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    # Exibindo grade e fundo sutil para melhorar a legibilidade
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.gca().set_facecolor('#f9f9f9')

    # Exibindo o gráfico
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    symbol = 'PETR4.SA'
    data = get_data_yfinance(symbol)
    data['target'] = data['close'].shift(-1)
    features = ['open', 'high', 'low', 'close', 'volume']

    # Adiciona indicadores técnicos
    data[f'EMA_14'] = Indicadores.media_movel_exponecial(data['close'], window=12)
    features.append(f'EMA_14')

    data['RSI'] = Indicadores().compute_RSI(data['close'], window=14)
    features.append('RSI')

    data['sto_rsi'] = Indicadores().get_stochastic_rsi(data['close'], window=14)
    features.append('sto_rsi')

    macd_histogram = Indicadores().compute_MACD(data['close'], short_window=12, long_window=26, signal_window=9)
    data['macd'] = macd_histogram
    features.append('macd')

    data.dropna(inplace=True)
    X = data[features]
    y = data['target']
    tscv = TimeSeriesSplit(n_splits=5)

    def print_model_results(model_name, model, scaler):
        results = evaluate_model(model, X, y, scaler, tscv)
        data[f'{model_name}_predictions'] = model.predict(scaler.transform(X))
        data['final_signal'] = np.where(data[f'{model_name}_predictions'] > data['close'], 1, -1)
        accuracy = check_signal_accuracy(data)
        backtest_results = backtest_strategy(data)
        
        print(f"\n{model_name} - Resultados:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Teste - MSE Médio: {results['test_avg_mse']:.4f}")
        print(f"Teste - MAE Médio: {results['test_avg_mae']:.4f}")
        print(f"Teste - R² Médio: {results['test_avg_r2']:.4f}")
        print(f"Treino - MSE Médio: {results['train_avg_mse']:.4f}")
        print(f"Treino - MAE Médio: {results['train_avg_mae']:.4f}")
        print(f"Treino - R² Médio: {results['train_avg_r2']:.4f}")
        print(f"Saldo Final da Carteira: {backtest_results['final_balance']:.2f}")
        print(f"Retorno Percentual da Carteira: {backtest_results['return_percentage']:.2f}%")
        return backtest_results['balance_history']

    best_rf_model, rf_scaler = GridSearchCV_RandomForestRegressor(X, y)
    best_lr_model, lr_scaler = GridSearchCV_LinearRegression(X, y)
    best_xgb_model, xgb_scaler = GridSearchCV_XGBoost(X, y)

    rf_balance_history = print_model_results("Random Forest", best_rf_model, rf_scaler)
    lr_balance_history = print_model_results("Linear Regression", best_lr_model, lr_scaler)
    xgb_balance_history = print_model_results("XGBoost", best_xgb_model, xgb_scaler)


    # Definindo a data inicial e a data máxima para o gráfico
    initial_date = datetime(2021, 1, 2)
    max_date = datetime(2023, 12, 28)

    # Valor inicial de investimento para o Buy and Hold
    initial_investment = 10000

    plot_trade_comparison(data, rf_balance_history, lr_balance_history, xgb_balance_history, initial_date, max_date, initial_investment)