import requests
import sys

url = "http://localhost:8000/predict"
file_path = "image.png"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    resp = requests.post(url, files=files)

print("Status:", resp.status_code)
print(resp.json())