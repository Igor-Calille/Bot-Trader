import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typing import List

from TreinarModelos.Test import Test


class View:
    def __init__(self):
        pass

    def graficos(stocks, portfolio_data):

        if len(stocks) == 0:
            print('Lista de stocks vazia')
            return
        else:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            for i, value in enumerate(stocks):
                accuracy = Test.check_signal_accuracy(value)
                initial_value = 10000
                final_value, backtest_results = Test.backtest_signals(value, initial_value)

                ax = axs[0, i]
                ax.plot(value.index, value['close'], label='Valor Real (Close)')
                ax.plot(value.index, value['predicted_price'], label='Valor Previsto', linestyle='--')
                ax.set_title(f'Comparação {value.name}')
                ax.set_xlabel('Data')
                ax.set_ylabel('Valor')
                ax.legend()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

                text_y = min(value['close']) - 3
                text_x = min(value.index) + pd.Timedelta(days=4)
                ax.text(text_x, text_y, f'Acurácia do sinal: {accuracy:.2f}\n'
                                        f'Backtest:\n'
                                        f'Valor inicial da carteira: ${initial_value}\n'
                                        f'Valor final da carteira: ${final_value}\n'
                                        f'Retorno do investimento:{(final_value- initial_value)/initial_value * 100:.2f}%',
                        wrap=True, horizontalalignment='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            # Gráfico do valor da carteira
            axs[1, 0].plot(portfolio_data['date'], portfolio_data['current_value'], label='Valor da Carteira', color='green')
            axs[1, 0].set_title('Valor da Carteira ao Longo do Tempo')
            axs[1, 0].set_xlabel('Data')
            axs[1, 0].set_ylabel('Valor da Carteira (USD)')
            axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axs[1, 0].legend()

            axs[1, 1].axis('off')  # Desativar o último subplot, se não for necessário

            plt.tight_layout()
            plt.show()
        
    def plot_investment_performance(predicted_stocks, initial_value, backtest_date):
        # Realiza o backtest e obtém os resultados
        backtest_results, count_trades = Test.backtest_signals_date_rpt(predicted_stocks, backtest_date, initial_capital=initial_value)
        final_value = backtest_results['current_value'].iloc[-1]

        # Informações do backtest
        print(f"Valor inicial da carteira: ${initial_value}")
        print(f"Valor final da carteira: ${final_value:.2f}")
        print(f"Quantidade de trades: {count_trades}")
        print(f"Retorno do investimento: {(final_value - initial_value) / initial_value * 100:.2f}%")

        # Configurações de estilo
        plt.style.use("ggplot")
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 linhas, 1 coluna

        # Gráfico de comparação entre valores reais e preditos
        axs[0].plot(predicted_stocks.index, predicted_stocks['close'], label='Valor Real (Close)')
        axs[0].plot(predicted_stocks.index, predicted_stocks['predicted_price'], label='Valor Predito', linestyle='--')
        axs[0].set_title('Comparação entre Valores Reais e Preditos')
        axs[0].set_xlabel('Data')
        axs[0].set_ylabel('Valor')
        axs[0].legend()

        # Anotação de texto no gráfico superior
        text_x = predicted_stocks.index[int(len(predicted_stocks) * 0.7)]
        text_y = min(predicted_stocks['close']) + (max(predicted_stocks['close']) - min(predicted_stocks['close'])) * 0.1
        axs[0].text(text_x, text_y, f"Valor inicial: ${initial_value}\nValor final: ${final_value:.2f}\nTrades: {count_trades}\nROI: {(final_value - initial_value) / initial_value * 100:.2f}%",
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Gráfico do valor da carteira ao longo do tempo
        axs[1].plot(backtest_results.index, backtest_results['current_value'], label='Valor da Carteira', color='green')
        axs[1].set_title('Valor da Carteira ao Longo do Tempo')
        axs[1].set_xlabel('Data')
        axs[1].set_ylabel('Valor da Carteira (USD)')
        axs[1].legend()

        # Ajuste do layout para evitar sobreposição de elementos
        plt.tight_layout()

        # Exibição do gráfico
        plt.show()

