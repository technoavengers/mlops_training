import requests

# API endpoint
url = "http://localhost:8002/predict/"

# Test payload
payload = {

      "transaction_id": 1,
      "customer_id": 2824,
      "product_id": 843,
      "product_name": "Fridge",
      "category": "Electronics",
      "quantity_sold": 3,
      "unit_price": 188.46,
      "transaction_date": "3/31/2024 21:46",
      "store_id": 3,
      "store_location": "Miami, FL",
      "inventory_level": 246,
      "reorder_point": 116,
      "reorder_quantity": 170,
      "supplier_id": 474,
      "supplier_lead_time": 8,
      "customer_age": 29,
      "customer_gender": "Other",
      "customer_income": 98760.83,
      "customer_loyalty_level": "Silver",
      "payment_method": "Credit Card",
      "promotion_applied": True,
      "promotion_type": "None",
      "weather_conditions": "Stormy",
      "holiday_indicator": False,
      "weekday": "Friday",
      "stockout_indicator": True,
      "forecasted_demand": 172

}

# Send POST request
response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Error parsing response:", e)
