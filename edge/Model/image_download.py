from google_images_download import google_images_download   #importing the library
from chromedriver_py import binary_path
from selenium import webdriver
response = google_images_download.googleimagesdownload() 
arguments = {"keywords":"downward dog AND yoga, dlank AND yoga, bridge AND yoga, <additional terms and variations, and even sanskrit names>","limit":50000,"print_urls":False, 'chromedriver': r"<location of your chromedrive.exe", "format": "jpg", "type": "photo"}   #creating list of arguments
paths = response.download(arguments)    #passing the arguments to the function
