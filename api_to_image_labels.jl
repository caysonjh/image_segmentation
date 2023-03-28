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

labels = project.label_generator()
labels = project.export_labels(download = True)

with open("labeled_images.txt", 'w') as outfile:
    for label in labels:
        outfile.write(f"{label['Labeled Data']}, {label['Label']['classifications'][0]['answer']['value']}\n")
        

"""