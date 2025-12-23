import pandas as pd
import requests
from io import StringIO

try:
    url = "https://www.investorgain.com/report/live-ipo-gmp/331/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    print(f"Fetching content from {url} with headers...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Success! Parsing HTML...")
        tables = pd.read_html(StringIO(response.text))
        
        if tables:
            print(f"Found {len(tables)} tables.")
            # Usually the first table is the main one
            df = tables[0]
            print("Columns:", df.columns)
            print(df.head())
        else:
            print("No tables found in the HTML.")
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")

except Exception as e:
    print(f"Error: {e}")
