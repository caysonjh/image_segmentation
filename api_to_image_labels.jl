using PyCall
using Images

py"""
import labelbox 
import urllib.request
from PIL import Image
import numpy as np 

LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGZkaTF0YnkxcXNjMDd4amgwbXZneTdoIiwib3JnYW5pemF0aW9uSWQiOiJjbDhhZHI3dmxhZTI4MDd4YjFmbmM1eDdlIiwiYXBpS2V5SWQiOiJjbGZkaWc3NjIwYzAwMDcxbjJhcXVoNHltIiwic2VjcmV0IjoiNjY5MTQyYjNjZTNmMThlYjllNzZmZmViY2M0MjY1OTEiLCJpYXQiOjE2NzkxMTYyMzAsImV4cCI6MjMxMDI2ODIzMH0.j-_7psuzORhMWZ5dZczAzx7y1LlM46vi1TX5HLA1-58"

lb = labelbox.Client(api_key=LB_API_KEY)

project = lb.get_project('cl97nfdyo0eh5071q5b8l3iom')
dataset = next(project.datasets())
data_rows = dataset.data_rows()
data = []

for data_row in data_rows:
    data_url = data_row.row_data
    data.append(data_url)


labels = project.label_generator()
labels = project.export_labels(download = True)

for label in labels: 
    data_url = labels[]

#image = labels[0]["Labeled Data"][:113]

import os
import requests
from PIL import Image
import io 

image_arrays = []
for index, url in enumerate(data): 
    response = requests.get(url)
    image_data = response.content

    image = Image.open(io.BytesIO(image_data))
    image.show()
    pixels = list(image.getdata())
    break
    


with open("labeled_images_final.txt", 'w') as outfile:
    for label in labels:
        outfile.write(f"{label['Labeled Data']}, {label['Label']['classifications'][0]['answer']['value']}\n")
        

"""