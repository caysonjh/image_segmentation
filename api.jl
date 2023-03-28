using Pkg

Pkg.add("PyCall")
using PyCall

py"""
import labelbox as lb
import random

LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGZkaTF0YnkxcXNjMDd4amgwbXZneTdoIiwib3JnYW5pemF0aW9uSWQiOiJjbDhhZHI3dmxhZTI4MDd4YjFmbmM1eDdlIiwiYXBpS2V5SWQiOiJjbGZkaWc3NjIwYzAwMDcxbjJhcXVoNHltIiwic2VjcmV0IjoiNjY5MTQyYjNjZTNmMThlYjllNzZmZmViY2M0MjY1OTEiLCJpYXQiOjE2NzkxMTYyMzAsImV4cCI6MjMxMDI2ODIzMH0.j-_7psuzORhMWZ5dZczAzx7y1LlM46vi1TX5HLA1-58"

client = lb.Client(api_key=LB_API_KEY)

dataset = client.get_dataset("cl8akly6j15ij073f2oor2i6s")

data_rows = dataset.data_rows()

data = []

for data_row in data_rows:
  data_url = data_row.row_data
  data.append(data_url)
"""

py"""
import os
import requests

# Create the img folder if it doesn't exist
if not os.path.exists("img"):
    os.makedirs("img")

# Download and save all images from the URLs stored in the data variable
for index, url in enumerate(data):
    response = requests.get(url)
    
    # Save the image in the img folder with a unique name
    with open(f"img/image_{index}.jpg", "wb") as f:
        f.write(response.content)
"""