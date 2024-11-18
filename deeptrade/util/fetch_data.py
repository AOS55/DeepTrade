import requests
import csv
from datetime import datetime

class AlphaVantageAPI:

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_equity_data(self, symbol, function='TIME_SERIES_DAILY', outputsize='full'):
        """
        Fetches equity data from the Alpha Vantage API.

        Parameters:
            function (str): The function specifying the type of data (e.g., 'TIME_SERIES_DAILY').
            symbol (str): The stock ticker symbol (e.g., 'IBM').
            outputsize (str): The amount of data to retrieve ('compact' or 'full'). Defaults to 'compact'.

        Returns:
            dict: The JSON response from the API containing the requested data.

        Raises:
            ValueError: If the API request fails or returns an error.
        """
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data or "Note" in data:
                raise ValueError(f"API returned an error: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
            return data
        else:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    def get_forex_data(self, from_currency, to_currency, function='FX_DAILY', outputsize='full'):
        """
        Fetches forex data from the Alpha Vantage API.

        Parameters:
            function (str): The function specifying the type of data (e.g., 'FX_DAILY').
            from_currency (str): The currency code to convert from (e.g., 'USD').
            to_currency (str): The currency code to convert to (e.g., 'EUR').
            outputsize (str): The amount of data to retrieve ('compact' or 'full'). Defaults to 'compact'.

        Returns:
            dict: The JSON response from the API containing the requested data.

        Raises:
            ValueError: If the API request fails or returns an error.
        """
        params = {
            'function': function,
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'apikey': self.api_key,
            'outputsize': outputsize
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data or "Note" in data:
                raise ValueError(f"API returned an error: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
            return data
        else:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    @staticmethod
    def save_to_csv(data, filename):
        """
        Saves the extracted data to a CSV file in the format DATETIME,price.

        Parameters:
            data (dict): The JSON response from the API containing the requested data.
            filename (str): The name of the CSV file to save the data to.
        """
        time_series = data.get('Time Series FX (Daily)', data.get('Time Series (Daily)', {}))
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['DATETIME', 'price']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for date, values in sorted(time_series.items()):
                datetime_str = f"{date}"
                price = values['4. close']
                writer.writerow({'DATETIME': datetime_str, 'price': price})

# Example usage:
if __name__ == "__main__":
    api_key = "PF5V2YY02M45DFSN"  # Replace with your API key
    api = AlphaVantageAPI(api_key)
    try:
        from_currency = 'USD'
        to_currency = 'GBP'
        data = api.get_forex_data(from_currency=from_currency, to_currency=to_currency)
        print(data.keys())
        api.save_to_csv(data, f'data/{from_currency}{to_currency}.csv')
        print("Data saved to output.csv")  # Process the data as needed
    except ValueError as e:
        print(e)
