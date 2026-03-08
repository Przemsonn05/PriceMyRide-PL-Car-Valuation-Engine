# src/utils.py
import requests

def get_current_eur_pln_rate():
    """Upload actual exchange rate EUR/PLN from NBP API."""
    url = "http://api.nbp.pl/api/exchangerates/rates/a/eur/?format=json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        rate = data['rates'][0]['mid']
        print(f"Exchange rate was loaded: {rate}")
        return rate
    except Exception as e:
        print(f"Uploading failed ({e}). Default value was set - 4.30.")
        return 4.30