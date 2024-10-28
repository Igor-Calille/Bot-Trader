import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import backtrader as bt
from datetime import datetime

def cross_val_evaluate_with_gridsearch(model, X, y, tscv, param_grid=None):
    mse_scores, mae_scores, r2_scores = [], [], []
    predictions = []

    if param_grid:
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        best_model = model.best_estimator_ if param_grid else model
        y_pred = best_model.predict(X_test)

        predictions.append((test_index, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    return avg_mse, avg_mae, avg_r2, predictions

def save_to_csv(data, rf_preds, lr_preds, avg_mse_rf, avg_mae_rf, avg_r2_rf, avg_mse_lr, avg_mae_lr, avg_r2_lr):
    results_df = pd.DataFrame({
        'Date': data['date'],
        'Real Prices': data['close'],
        'Random Forest Predictions': rf_preds,
        'Linear Regression Predictions': lr_preds,
        'Signal ML RF': data['signal_ml_rf'],
        'Signal ML LR': data['signal_ml_lr']
    })

    metrics_data = {
        'Metric': ['Avg MSE', 'Avg MAE', 'Avg R²'],
        'Random Forest': [avg_mse_rf, avg_mae_rf, avg_r2_rf],
        'Linear Regression': [avg_mse_lr, avg_mae_lr, avg_r2_lr]
    }
    metrics_df = pd.DataFrame(metrics_data)

    results_df.to_csv('forecasting_results.csv', index=False)
    metrics_df.to_csv('forecasting_results_metrics.csv', index=False)
    print("Dados salvos em forecasting_results.csv e forecasting_results_metrics.csv.")

def run_backtesting(data, signal_column):
    class PandasData(bt.feeds.PandasData):
        lines = ('signal_ml',)
        params = (('signal_ml', -1),)

    class MLStrategy(bt.Strategy):
        params = (('start_date', datetime(202, 8, 30)), ('risk_per_trade', 1.0),)

        def __init__(self):
            self.start_trading = False

        def next(self):
            current_date = self.data.datetime.date(0)
            if current_date >= self.params.start_date.date():
                self.start_trading = True

            if self.start_trading:
                if self.data.signal_ml[0] == 1 and self.broker.get_cash() > 0:
                    size = self.broker.get_cash() / self.data.close[0]
                    self.buy(size=size)
                elif self.data.signal_ml[0] == -1 and self.position:
                    self.sell(size=self.position.size)

    # Converter a coluna 'date' para datetime, se necessário
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy)
    cerebro.adddata(PandasData(dataname=data, datetime='date', signal_ml=signal_column))
    cerebro.broker.set_cash(10000)
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.plot()

def main_comparative_analysis_tscv(X, y, data):
    tscv = TimeSeriesSplit(n_splits=5)
    rf_model = RandomForestRegressor(random_state=42)
    lr_model = LinearRegression()

    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    lr_param_grid = {'fit_intercept': [True, False]}

    rf_mse, rf_mae, rf_r2, rf_preds = cross_val_evaluate_with_gridsearch(rf_model, X, y, tscv, rf_param_grid)
    lr_mse, lr_mae, lr_r2, lr_preds = cross_val_evaluate_with_gridsearch(lr_model, X, y, tscv, lr_param_grid)

    print("Random Forest Regressor:")
    print(f"  Avg MSE: {rf_mse:.4f}")
    print(f"  Avg MAE: {rf_mae:.4f}")
    print(f"  Avg R²: {rf_r2:.4f}\n")

    print("Linear Regression:")
    print(f"  Avg MSE: {lr_mse:.4f}")
    print(f"  Avg MAE: {lr_mae:.4f}")
    print(f"  Avg R²: {lr_r2:.4f}\n")

    rf_full_preds = np.full(y.shape, np.nan)
    lr_full_preds = np.full(y.shape, np.nan)

    for idx, pred in rf_preds:
        rf_full_preds[idx] = pred

    for idx, pred in lr_preds:
        lr_full_preds[idx] = pred

    data['signal_ml_rf'] = np.where(rf_full_preds > data['close'], 1, -1)
    data['signal_ml_lr'] = np.where(lr_full_preds > data['close'], 1, -1)

    save_to_csv(data, rf_full_preds, lr_full_preds, rf_mse, rf_mae, rf_r2, lr_mse, lr_mae, lr_r2)

    run_backtesting(data, 'signal_ml_rf')
    run_backtesting(data, 'signal_ml_lr')

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

if __name__ == '__main__':
    symbol = 'AAPL'
    data = get_data_yfinance(symbol)
    data['target'] = data['close'].shift(-1).fillna(method='ffill')
    features = ['open', 'high', 'low', 'close', 'volume']
    X = data[features].values
    y = data['target'].values

    main_comparative_analysis_tscv(X, y, data)
