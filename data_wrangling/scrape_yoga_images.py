import os
import pprint
import json
import hashlib
from collections import defaultdict
from io import BytesIO
import urllib

# pip install azure-cognitiveservices-search-imagesearch
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image

API_KEY = os.environ['BING_SEARCH_API_KEY']
DATA_DIR = './data'
DATA_WRANGLE_DIR = './data_wrangling'
IMAGE_BASE_DIR = os.path.join(DATA_DIR, 'out')
QUERIES_FILE = os.path.join(DATA_WRANGLE_DIR, 'queries.json')

NUM_RESULTS = 500
SINGLE_QUERY_MAX_RESULTS = 150

def get_hash(data):
    return hashlib.md5(data)

def load_json(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)

def load_queries():
    return load_json(QUERIES_FILE)

def load_current_data():
    class_counts = defaultdict(int)
    hashes = set()
    for class_dir in os.listdir(IMAGE_BASE_DIR):
        class_path = os.path.join(IMAGE_BASE_DIR, class_dir)
        for img in os.listdir(class_path):
            file_path = os.path.join(class_path, img)
            new_hash = get_hash(open(file_path, 'rb').read())

            if new_hash in hashes:
                raise ValueError('File {file_path} hash collision')

            class_counts[class_dir] += 1
            hashes.add(new_hash)

    return dict(class_counts), hashes

def download_url(url):
    return urllib.request.urlopen(url).read()

def download_images_from_query_text(client, class_name, query_string, hashes, offset):

    def images_generator():
        curr_offset = 0
        while curr_offset < NUM_RESULTS:
            new_count = min([NUM_RESULTS - curr_offset, SINGLE_QUERY_MAX_RESULTS]) 
            response = client.images.search(query_string, count=new_count, offset=curr_offset)
            if not response.value:
                raise ValueError('Query {} did not return any results')
            curr_offset += len(response.value)

            for img in response.value:
                yield img

    count = 0
    for img in images_generator():
        try:
            img_bytes = download_url(img.content_url)
        except urllib.request.HTTPError as e:
            print(f'Received error code {e.code} for url {img.content_url}')
            continue
        except urllib.error.URLError:
            print(f'URL {img.content_url} could not be found. Bing, you liar...')

        new_img_hash = get_hash(img_bytes)
        if new_img_hash in hashes:
            print('Image at url {img.content_url} for class {class_name} already in dataset, skipping...')
            continue
        
        hashes.add(new_img_hash)

        dest_dir = os.path.join(IMAGE_BASE_DIR, class_name, f'{offset + count}.jpg')
        try:
            img = Image.open(BytesIO(img_bytes))
            img.convert('RGB').save(dest_dir, "JPEG")
        except OSError:
            print(f'There was an error while retrieving and image from url {img.content_url}')
        count += 1
    
    return count
    

def main():
    # load the counts and hashes of the current images
    print('Loading current data...')
    class_counts, hashes = load_current_data()
    print(f'Loaded {len(hashes)} hashes\n')
    print(f'Class counts: {pprint.pformat(class_counts)}\n')

    # load the query information
    print('Loading queries...')
    queries = load_queries()
    print(f'Queries: {pprint.pformat(queries)}\n')

    # initialize the search client
    client = ImageSearchAPI(CognitiveServicesCredentials(API_KEY))


    # iterate over the different classes and retrieve the images
    for class_name, class_queries in queries.items():
        print('Downloading {class_name} images')
        # first just do the plaintext query with "yoga pose" appended

        all_queries = [class_queries['plaintext_name'] + " yoga pose"] + class_queries['other_queries']
        for q in all_queries:
            print(f'Running query "{q}"')
            count = download_images_from_query_text(client,
                class_name, 
                q, 
                hashes, 
                class_counts[class_name]
            )

            class_counts[class_name] += count
    
    print(f'Done! Class Counts: {pprint.pformat(class_counts)}')

    




if __name__ == '__main__':
    main()