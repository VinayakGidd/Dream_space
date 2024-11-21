import requests

GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CX = "YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID"

def search_products_google(product_query):
    try:
        # Use Google Custom Search API to find products
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": product_query,
                "num": 5,  # Limit results to 5
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Google API error: {response.json()}")

        products = response.json().get("items", [])
        return products

    except Exception as e:
        raise ValueError(f"Error during Google product search: {e}")
