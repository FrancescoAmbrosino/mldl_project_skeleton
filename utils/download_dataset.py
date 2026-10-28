import os
import shutil
import urllib.request
import zipfile

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

# data/ at the repo root, whatever folder the script is launched from (data/* is git-ignored)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
ZIP_PATH = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
DATASET_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200')


def download():
    # Skip if already downloaded (like wget -nc). Download to .part and rename at the end,
    # so an interrupted download is never mistaken for a complete zip
    if os.path.exists(ZIP_PATH):
        return
    print(f'Downloading {URL}', flush=True)
    urllib.request.urlretrieve(URL, ZIP_PATH + '.part')
    os.rename(ZIP_PATH + '.part', ZIP_PATH)


def extract():
    # Skip if already extracted (like unzip -n). Same trick: extract to a temporary folder
    # and move the dataset into place only when the extraction is complete
    if os.path.isdir(DATASET_DIR):
        return
    print(f'Extracting {ZIP_PATH}', flush=True)
    tmp_dir = os.path.join(DATA_DIR, 'extracting')
    shutil.rmtree(tmp_dir, ignore_errors=True)  # leftovers of an interrupted run
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(tmp_dir)
    os.rename(os.path.join(tmp_dir, 'tiny-imagenet-200'), DATASET_DIR)
    shutil.rmtree(tmp_dir)


def reorganize_val():
    # ImageFolder wants one folder per class: copy val/images/<file> to val/<class>/<file>.
    # Skip if val/ is already reorganized (images/ is deleted at the end)
    val_dir = os.path.join(DATASET_DIR, 'val')
    images_dir = os.path.join(val_dir, 'images')
    if not os.path.isdir(images_dir):
        return
    print('Reorganizing val/ into class folders', flush=True)
    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.copyfile(os.path.join(images_dir, fn), os.path.join(val_dir, cls, fn))
    shutil.rmtree(images_dir)


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    download()
    extract()
    reorganize_val()
    print(f'Dataset ready in {DATASET_DIR}')
