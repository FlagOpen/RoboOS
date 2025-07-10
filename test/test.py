import requests

url = "http://127.0.0.1:5000/publish_task"

data = {"refresh": True, "task": "Navigate to P1, pick up apple"}
response = requests.post(url, json=data)
