import yfinance as yf
from TreinarModelos.Merge import BOTS

def main():
    symbols = ['AAPL', 'MSFT', 'VALE3.SA', 'PETR4.SA']

    for symbol in symbols:
        data = get_data_yfinance(symbol)
        
        data_ML = BOTS.BOT_main(
            data,
            'LinearRidge_GridSearchCV',
            media_movel_exponencial=[12],
            rsi=True,
            MACD=True
        )


def get_data_yfinance(symbol="AAPL"):
    from_date = "2015-01-02"
    to_date = "2023-12-28"

    # Obter stocks
    data = yf.download(symbol, start=from_date, end=to_date)

    if data.empty:
        print('Falha no acesso dos dados históricos.')
    else:
        print(f'Dados obtidos, total de linhas: {len(data)}')

    # Processar os dados
    data.reset_index(inplace=True)
    data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})
    #data = data.rename(columns={'Datetime': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'})

    return data


if __name__ == '__main__':
    main()

