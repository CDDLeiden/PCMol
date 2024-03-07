import os, time
from re import S
from urllib import request
import zipfile
import argparse
from pcmol.config import dirs


DATASET_URL_LITE = r"https://surfdrive.surf.nl/files/index.php/s/Gqy5vPOYJUHVaU7/download\?path\=%2F\&files\="


def download_protein_data(pid, full_data=False, unzip=True, status="", data_url=None):
    """Downloads the processed protein sequence from online storage"""

    print(
        f"{status} Downloading AlphaFold embeddings for {pid} to <{dirs.ALPHAFOLD_DIR}>..."
    )

    if data_url is None:
        url = DATASET_URL_LITE
        data_url = url + f"{pid}.zip"

    if os.path.isdir(os.path.join(dirs.ALPHAFOLD_DIR, pid)):
        print(f"{status} Protein sequence {pid} already downloaded.")
        return True

    print(f"{status} Downloading from {data_url}...")
    local_file = os.path.join(dirs.ALPHAFOLD_DIR, f"{pid}.zip")
    # open(local_file, 'a').close()
    try:
        # request.urlretrieve(data_url, local_file) # This method is deprecated (after update to surfdrive)
        os.system(f"wget -O {local_file} {data_url}")
    except Exception as e:
        os.remove(local_file)
        print(f"Files for protein {pid} could not be downloaded.", e)
        return False

    try:
        if unzip:
            unzip_file(local_file)
        return True
    except:
        print(f"Could not unzip file {pid}.")
        return False


def unzip_file(filename, remove_original=True):
    """Unzips a file"""

    with zipfile.ZipFile(filename, "r") as zip_ref:
        target_dir = os.path.join(dirs.ALPHAFOLD_DIR, filename.split(".")[0])
        os.makedirs(target_dir, exist_ok=True)
        zip_ref.extractall(target_dir)
    if remove_original:
        os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pid", type=str, default=None)
    parser.add_argument("-f", "--pid_file", type=str, default=None)
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-x", "--full", action="store_true")
    parser.add_argument("-s", "--subset", type=str, default=None)
    args = parser.parse_args()

    print(f'{"*"*30}\nDownloading foldedPapyrus data...\n{"*"*30}\n')

    # Single protein
    if args.pid:
        status = download_protein_data(args.pid, full_data=args.full)
        if status:
            print(f"\nSuccessfully downloaded data for protein {args.pid}")

    # Proteins from a file
    elif args.pid_file:

        if not args.subset:
            subset_name = os.path.split(args.pid_file.split(".")[0])[-1]
        else:
            subset_name = args.subset

        ALPHAFOLD_DIR = os.path.join(dirs.ALPHAFOLD_DIR, subset_name)

        os.makedirs(ALPHAFOLD_DIR, exist_ok=True)
        unavailable, processed = [], 0
        with open(args.pid_file, "r") as f:
            lines = f.readlines()
            print(f"Downloading {len(lines)} proteins listed in {args.pid_file}...\n")
            for i, line in enumerate(lines):
                accession = line.strip().split("_")[0]
                progress = f"{i+1}/{len(lines)}"

                status = download_protein_data(
                    accession, full_data=args.full, status=progress
                )
                if not status:
                    unavailable.append(accession)
                else:
                    processed += 1

                # To prevent serving too many requests to the server
                time.sleep(1)

        formatted = "\n".join(unavailable)
        print(f'{"*"*50} \nDownloaded data for {processed}/{len(lines)} proteins.')
        print(f'The following proteins were not found: \n{formatted}\n{"*"*50}')

    # Download all proteins
    elif args.all:
        with open("utils/processed_pids.csv", "r") as f:
            for pid in f.readlines():
                download_protein_data(pid.strip(), full_data=args.full)
