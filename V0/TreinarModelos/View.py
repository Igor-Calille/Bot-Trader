import matplotlib.pyplot as plt
import numpy as np
from typing import List

from TreinarModelos.Test import Test


class View:
    def __init__(self):
        pass

    def graficos(stocks):

        if len(stocks) == 0:
            print('Lista de stocks vazia')
            return
        else:
            plt.figure(figsize=(12, 6))
            for i, value in enumerate(stocks):
                accuracy = Test.check_signal_accuracy(value)
                initial_value = 10000
                final_value, backtest_results = Test.backtest_signals(value,initial_value)

                plt.subplot(1, 2, i+1)
                plt.plot(value.index, value['close'], label='Valor Real (Close)')
                plt.plot(value.index, value['predicted_price'], label='Valor Previsto', linestyle='--') 
                plt.title('Comparação entre Valores Reais e Preditos')
                plt.xlabel('Data')
                plt.ylabel('Valor')
                plt.legend()

                
                #text_y = min(value['close']) - (max(value['close']) - min(value['close'])) * 0.1
                #text_x = min(value.index)
                text_y = min(value['close']) - 3
                text_x = min(value.index) + 4
                plt.text(text_x, text_y,f'Acurácia do sinal: {accuracy:.2f}\n Backtest:\nValor inicial da carteira: ${initial_value}\nValor final da carteira: ${final_value}\nRetorno do investimento:{(final_value- initial_value)/initial_value * 100:.2f}%', wrap=True, horizontalalignment='center', fontsize=12)
            plt.tight_layout()
            plt.show()

