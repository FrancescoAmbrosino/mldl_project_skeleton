import os
import shutil
import urllib.parse
import urllib.request

# --- Dataset settings: to reuse the script for another dataset, change only this block ---

# Archive to download: zip or tar (.tar, .tar.gz, ...), the format comes from the extension
URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

# Name to save the archive under: the last part of the URL. Write it by hand
# if the URL does not end with a file name
ARCHIVE_NAME = os.path.basename(urllib.parse.urlparse(URL).path)

# Top-level folder inside the archive: the dataset ends up in DATA_DIR/EXTRACTED_DIR
EXTRACTED_DIR = 'tiny-imagenet-200'

# data/ at the repo root, found from this file's position (utils/ is one level down),
# so it works whatever folder the script is launched from (data/* is git-ignored)
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

# ImageFolder wants one folder per class. A split whose images sit in one flat folder,
# with an annotations file giving each image's class, is copied to <split>/<class>/<file>.
# REORG_SPLIT = None skips this step, for datasets already split into class folders
REORG_SPLIT = 'val'                        # split folder, inside the dataset folder
REORG_IMAGES = 'images'                    # flat image folder, inside the split
REORG_ANNOTATIONS = 'val_annotations.txt'  # annotations file, inside the split
REORG_SEP = '\t'                           # column separator of the annotations
REORG_FILE_COL, REORG_CLASS_COL = 0, 1     # columns holding the file name and the class

# --- End of the dataset settings ---

ARCHIVE_PATH = os.path.join(DATA_DIR, ARCHIVE_NAME)
DATASET_DIR = os.path.join(DATA_DIR, EXTRACTED_DIR)


def download():
    # Skip if already downloaded (like wget -nc). Download to .part and rename at the end,
    # so an interrupted download is never mistaken for a complete archive
    if os.path.exists(ARCHIVE_PATH):
        return
    print(f'Downloading {URL}', flush=True)
    urllib.request.urlretrieve(URL, ARCHIVE_PATH + '.part')
    os.rename(ARCHIVE_PATH + '.part', ARCHIVE_PATH)


def extract():
    # Skip if already extracted (like unzip -n). Same trick: extract to a temporary folder
    # and move the dataset into place only when the extraction is complete
    if os.path.isdir(DATASET_DIR):
        return
    print(f'Extracting {ARCHIVE_PATH}', flush=True)
    tmp_dir = os.path.join(DATA_DIR, 'extracting')
    shutil.rmtree(tmp_dir, ignore_errors=True)  # leftovers of an interrupted run
    # Tar only: the 'data' filter refuses links and paths that point outside tmp_dir
    # (zip extraction already drops them, and its unpacker takes no filter argument)
    tar_only = {} if ARCHIVE_NAME.endswith('.zip') else {'filter': 'data'}
    shutil.unpack_archive(ARCHIVE_PATH, tmp_dir, **tar_only)
    extracted = os.path.join(tmp_dir, EXTRACTED_DIR)
    if not os.path.isdir(extracted):
        raise FileNotFoundError(f'no {EXTRACTED_DIR}/ in the archive, which holds '
                                f'{sorted(os.listdir(tmp_dir))}: fix EXTRACTED_DIR')
    os.rename(extracted, DATASET_DIR)
    shutil.rmtree(tmp_dir)


def reorganize_split():
    # Copy <split>/<images>/<file> to <split>/<class>/<file>, then delete the flat folder.
    # Skip if there is nothing to reorganize or it is already done (the flat folder is gone)
    if REORG_SPLIT is None:
        return
    split_dir = os.path.join(DATASET_DIR, REORG_SPLIT)
    images_dir = os.path.join(split_dir, REORG_IMAGES)
    if not os.path.isdir(images_dir):
        return
    print(f'Reorganizing {REORG_SPLIT}/ into class folders', flush=True)
    with open(os.path.join(split_dir, REORG_ANNOTATIONS)) as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip('\r\n').split(REORG_SEP)
            fn, cls = cols[REORG_FILE_COL], cols[REORG_CLASS_COL]
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
            shutil.copyfile(os.path.join(images_dir, fn), os.path.join(split_dir, cls, fn))
    shutil.rmtree(images_dir)


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    download()
    extract()
    reorganize_split()
    print(f'Dataset ready in {DATASET_DIR}')
