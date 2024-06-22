import numpy as np

class Test:
    def __init__ (self):
        pass

    #Função para calcular a acurácia do sinal de trading
    def check_signal_accuracy(data):
        # Calcular a mudança de preço do dia seguinte
        data['price_change'] = data['close'].shift(-1) - data['close']

        # Definir condições para um sinal correto de compra ou venda
        conditions = [
            (data['final_signal'] > 0) & (data['price_change'] > 0),  # Compra seguida de aumento de preço
            (data['final_signal'] < 0) & (data['price_change'] < 0),  # Venda seguida de queda de preço
        ]
        choices = [1, 1]  # Ambos são sinais corretos, independente de serem compra ou venda
        data['correct_signal'] = np.select(conditions, choices, default=0)

        # Calcular a acurácia do sinal
        correct_signals = data['correct_signal'].sum()
        total_signals = np.count_nonzero(data['final_signal'])  # Conta todos os sinais emitidos, ignorando zeros
        accuracy = correct_signals / total_signals if total_signals > 0 else 0  # Evita divisão por zero
        return accuracy
    
    def backtest_signals(data, initial_capital=10000, risk_per_trade=1.0):
        cash = initial_capital
        position = 0
        portfolio_value = []

        for index, row in data.iterrows():
            buy_price = row['close'] * 1.01  # Inclui slippage
            sell_price = row['close'] * 0.99  # Inclui slippage
            amount_to_risk = cash * risk_per_trade

            if row['action'] == 'Buy' and cash > 0:
                shares_to_buy = amount_to_risk / buy_price
                position += shares_to_buy
                cash -= shares_to_buy * buy_price

            elif row['action'] == 'Sell' and position > 0:
                cash += position * sell_price
                position = 0

            current_value = cash + position * row['close']
            portfolio_value.append(current_value)

        data['Portfolio Value'] = portfolio_value
        return data['Portfolio Value'].iloc[-1], data

    def backtest_signals_SL_TP(data, initial_capital=10000, risk_per_trade=0.02, stop_loss_percent=0.05, take_profit_percent=0.10):
        cash = initial_capital
        position = 0
        portfolio_value = []
        entry_price = 0

        for index, row in data.iterrows():
            buy_price = row['close'] * 1.01
            sell_price = row['close'] * 0.99
            stop_loss_price = entry_price * (1 - stop_loss_percent)
            take_profit_price = entry_price * (1 + take_profit_percent)

            if row['action'] == 'Buy' and cash > 0:
                position = (cash * risk_per_trade) / buy_price
                cash -= position * buy_price
                entry_price = buy_price
            elif row['action'] == 'Sell' and position > 0:
                if sell_price <= stop_loss_price or sell_price >= take_profit_price:
                    cash += position * sell_price
                    position = 0

            current_value = cash + position * row['close']
            portfolio_value.append(current_value)

        data['Portfolio Value'] = portfolio_value
        return data['Portfolio Value'].iloc[-1], data




