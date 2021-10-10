import pandas as pd
from PIL import Image as pilim
from imutils import paths
import numpy as np
import warnings
import requests
from tqdm import tqdm
import tarfile
from pandas.core.common import SettingWithCopyWarning
import shutil
import os


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_image_dims(file_path):
    im = pilim.open(file_path)
    # returns (w,h) after rotation-correction
    return im.size if im._getexif().get(274,0) < 5 else im.size[::-1]


def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def unzip(path: str):
    file = tarfile.open(path)
    file.extractall('.')
    file.close()


def main():
    cols = ["filename", "x_i", "y_i", "w_i", "h_i", "b_i"]
    master_df = pd.read_csv("https://raw.githubusercontent.com/gulvarol/grocerydataset/master/annotations.csv", 
                        names=cols)
    master_df['b_i'] = 0
    master_df = master_df.rename(columns={'b_i': 'class', 'x_i': 'xmin', 'y_i': 'ymin',
                                        'w_i': 'xmax', 'h_i': 'ymax'})

    print('Downloading and unzipping main tarball...')

    download('https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz', 'ShelfImages.tar.gz')
    unzip('ShelfImages.tar.gz')

    print('Changing labels to appropriate format...')

    train_images = list(paths.list_images("ShelfImages/train"))
    test_images = list(paths.list_images("ShelfImages/test"))
    train_image_names = [image_path.split("/")[-1] for image_path in train_images]
    test_image_names = [image_path.split("/")[-1] for image_path in test_images]

    # Create two different dataframes from train and test sets
    train_df = master_df[master_df["filename"].isin(train_image_names)]
    test_df = master_df[~master_df["filename"].isin(train_image_names)]

    # Add absolute paths
    train_df.loc[:,"filename"] = train_df.loc[:,"filename"].map(lambda x: "ShelfImages/train/" + x)
    test_df.loc[:,"filename"] = test_df.loc[:,"filename"].map(lambda x: "ShelfImages/test/" + x)

    # Get image dimensions
    train_df.loc[:,"width"] = train_df.loc[:,"filename"].map(lambda x: get_image_dims(x)[0])
    train_df.loc[:,"height"] = train_df.loc[:,"filename"].map(lambda x: get_image_dims(x)[1])
    test_df.loc[:,"width"] = test_df.loc[:,"filename"].map(lambda x: get_image_dims(x)[0])
    test_df.loc[:,"height"] = test_df.loc[:,"filename"].map(lambda x: get_image_dims(x)[1])

    # Filtering bad bboxes out, e.g. C1_P01_N1_S2_1.JPG
    train_df = train_df[train_df['ymax']/train_df['height'] <= 1]
    test_df = test_df[test_df['ymax']/test_df['height'] <= 1]

    train_df.to_csv('ShelfImages/train.csv', encoding='utf-8', index=False)
    test_df.to_csv('ShelfImages/val.csv', encoding='utf-8', index=False)

    train_df = pd.read_csv('ShelfImages/train.csv')
    test_df = pd.read_csv('ShelfImages/val.csv')

    train_df['filename'] = train_df['filename'].map(lambda x: x.split('/')[-1])
    test_df['filename'] = test_df['filename'].map(lambda x: x.split('/')[-1])

    train_df['class'] = 'Product'
    test_df['class'] = 'Product'

    print(f'Total: {len(master_df)}, train and val splits: {len(train_df)} and {len(test_df)}')

    prev_image = ''
    image_id = -1
    for i in range(len(test_df)):
        if prev_image != test_df['filename'][i]:
            prev_image = test_df['filename'][i]
            image_id += 1
        test_df.loc[i,'image_id'] = image_id

    prev_image = ''
    image_id = -1
    for i in range(len(train_df)):
        if prev_image != train_df['filename'][i]:
            prev_image = train_df['filename'][i]
            image_id += 1
        train_df.loc[i,'image_id'] = image_id

    train_df = train_df[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']]
    test_df = test_df[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']]

    train_df.to_csv('ShelfImages/train.csv', encoding='utf-8', index=False)
    test_df.to_csv('ShelfImages/val.csv', encoding='utf-8', index=False)

    print("Moving and removing files from directories...")

    source = "ShelfImages/test"
    destination = "ShelfImages/train"
    files_list = os.listdir(source)
    for files in files_list:
        shutil.move(source + '/' + files, destination)

    if len(os.listdir('ShelfImages/test')) == 0: # Check if the folder is empty
        shutil.rmtree('ShelfImages/test') # If so, delete it

    os.rename("ShelfImages/train", "ShelfImages/images")

    print('Data prep complete.')

main()