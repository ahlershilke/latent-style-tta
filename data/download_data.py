import argparse
import tarfile
from zipfile import ZipFile
import gdown
import os
import shutil
import kagglehub
import PIL
from PIL import Image


"""
This code was taken and modified from: 
 https://github.com/VoErik/domain-generalization/blob/main/domgen/data/download_data.py
"""


# the downloading pacs + stage_path + download and extract functions (with slight modifications) are taken directly from:
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py


def download_pacs(data_dir):
    if data_dir is None:
        data_dir = os.getcwd()

    full_path = ask_for_download_location(os.path.join(data_dir, "PACS"))

    kagglehub_path = kagglehub.dataset_download("ma3ple/pacs-dataset")
    join_paths(kagglehub_path, full_path)

    kfold_path = os.path.join(os.path.dirname(full_path), "kfold")
    if os.path.exists(kfold_path):
        join_paths(kfold_path, full_path)


def download_vlcs(data_dir):
    if data_dir is None:
        data_dir = os.getcwd()

    full_path = ask_for_download_location(os.path.join(data_dir, "VLCS"))

    kagglehub_path = kagglehub.dataset_download("iamjanvijay/vlcsdataset")
    join_paths(kagglehub_path, full_path)

    kfold_path = os.path.join(os.path.dirname(full_path), "kfold")
    if os.path.exists(kfold_path):
        join_paths(kfold_path, full_path)


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def join_paths(source, destination):
    os.makedirs(destination, exist_ok=True)
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source doesn't exist: {source}")
    
    print(f"\n[INFO] Moving from '{source}' to '{destination}'")
    
    for item in os.listdir(source):
        src = os.path.join(source, item)
        dst = os.path.join(destination, item)
    
        if os.path.exists(dst):
            print(f"[WARN] Destination '{dst}' already exists, will rewrite.")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
                
        shutil.move(src, dst)
    
    try:
        shutil.rmtree(source)
        print(f"[OK] Join complete.")
    except OSError:
        print(f"[WARN] Source folder couldn't be deleted: {source}")


def ask_for_download_location(default_dir):
    print(f"\nDefault download folder: {default_dir}")
    response = input("Would you like your data in another folder? (y/n): ").strip().lower()
    
    if response == 'y':
        custom_dir = input("Please insert full path to target folder (will be created if doesn't exist): ").strip()
        custom_dir = os.path.expanduser(custom_dir)
        os.makedirs(custom_dir, exist_ok=True)

        return os.path.abspath(custom_dir)
    
    if response == 'n':
        print(f"Using default folder: {default_dir}")
        default_dir = os.path.expanduser(default_dir)
        os.makedirs(default_dir, exist_ok=True)
        
        return os.path.abspath(default_dir)


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./datasets')
    parser.add_argument('--download_vlcs', action='store_true', default=False)
    parser.add_argument('--download_pacs', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args()

    if args.all:
        # insert all datasets here
        args.download_pacs = True
        args.download_vlcs = True
    if args.download_pacs:
        download_pacs(args.datadir)
    if args.download_vlcs:
        download_vlcs(args.datadir)