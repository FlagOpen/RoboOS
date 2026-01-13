#!/usr/bin/env python3

import os

if __name__ == '__main__':
    dataset_url = 'https://storage.googleapis.com/niantic-lon-static/research/ace/wayspots.tar.gz'
    dataset_file = dataset_url.split('/')[-1]

    print("\n#####################################################################")
    print("# Please make sure to check this dataset's license before using it! #")
    print("# https://nianticlabs.github.io/ace/wayspots.html                   #")
    print("#####################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    print("Downloading and unzipping data...")
    os.system('wget ' + dataset_url)
    os.system('tar -xzf ' + dataset_file)
    os.system('rm ' + dataset_file)
    print("Done.")
