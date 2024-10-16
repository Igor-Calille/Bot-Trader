import requests

url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=OQROSK31QKAOHXMQ'
r = requests.get(url)
data = r.json()

print(data)