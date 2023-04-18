import os
from pathlib import Path
from tqdm import tqdm
from magma.datasets.convert_datasets import convert_dataset

def get_ds_iterator(tmp_path):
    for image in tmp_path.glob("*.jpg"):
        try:
            with open(image.with_suffix('.txt'), "r") as f:
                yield (image, {"captions" : f.readlines()[0]})
        except:
            pass

if __name__ == "__main__":
    original_datasets = '/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data'
    destination_datasets = '/gpfs/alpine/csc499/proj-shared/magma/LAION-400m-webdataset'

    original_datasets = Path(original_datasets)
    destination_datasets = Path(destination_datasets)

    os.makedirs(destination_datasets / 'tmp', exist_ok=True)

    seen_archives = [p.stem for p in (destination_datasets/'images').glob('*')]

    for path in tqdm(
            original_datasets.glob("*.tar"),
            desc=f"loading dataset tar form {original_datasets}",
        ):

        archive_number = path.stem

        if archive_number in seen_archives:
            continue

        tmp_path = destination_datasets / 'tmp' / archive_number
        
        os.makedirs(tmp_path, exist_ok=True)
        os.system(f"tar -xf {path} -C {tmp_path}")

        ds_iterator = get_ds_iterator(tmp_path)

        convert_dataset(
            destination_datasets,
            dir_size=10000,
            mode="mv",
            ds_iterator=ds_iterator,
            dataset_number=archive_number
        )
        
        os.system(f'rm -rf {tmp_path}')
    
    os.system(f"rm -rf {destination_datasets / 'tmp'}")

