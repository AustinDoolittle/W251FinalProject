from google_images_download import google_images_download   #importing the library
from chromedriver_py import binary_path
from selenium import webdriver
response = google_images_download.googleimagesdownload() 
arguments = {"keywords":"~Kumbhakasana","limit":50000,"print_urls":False, 'chromedriver': r"C:\Users\steph\Desktop\Berkeley\MachineLearn\projects\chromedriver_win32\chromedriver", "format": "jpg", "type": "photo"}   #creating list of arguments
paths = response.download(arguments)    #passing the arguments to the function