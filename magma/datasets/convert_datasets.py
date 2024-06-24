from PIL import Image
from PIL import UnidentifiedImageError
import os
import json
from pathlib import Path
from tqdm import tqdm
import shutil


def save_to_jsons(data_list, target_dir, starting_idx=0):
    pbar = tqdm(
        enumerate(data_list), desc=f"saving {len(data_list)} jsons to {str(target_dir)}"
    )
    for k, data in pbar:
        filename = Path(target_dir) / Path(f"{k+starting_idx}.json")
        with open(filename, "w") as f:
            json.dump(data, f)

    return None


def save_images(img_list, target_dir, mode="mv"):
    #print(f'img_list: {img_list}')
    for img_path in tqdm(
        img_list,
        desc=f"saving {len(img_list)} images (mode={mode}) to {str(target_dir)}",
    ):
        if mode == "mv":
            shutil.move(img_path, target_dir)
        elif mode == "cp":
            shutil.copy(img_path, target_dir)


def convert_dataset(
    data_dir,
    dir_size=10000,
    hash_fn=None,
    mode="cp",
    ds_iterator=None,
    dataset_number=0
):
    """
    Builds a dataset directory in our standard format. ds_iterator should return data of the form
    image_path, {"captions": [...], "metadata": {...}, }, where image_path should be a Path object, captions should map to a list of strings
    and metadata can contain any custom data about the image. If a hash_fn is specified (such as phash), the image hash gets saved in metadata.
    """

    data_dir = Path(data_dir)

    # folders for images and corresponding data which is stored in a json file for each image
    os.makedirs(data_dir / "images", exist_ok=True)
    os.makedirs(data_dir / "image_data", exist_ok=True)

    img_data_list = []
    img_path_list = []
    save_img_dir = data_dir / "images" / str(dataset_number)
    save_data_dir = data_dir / "image_data" / str(dataset_number)
    num_img_dirs = 0

    # save the new locations of all img files in case some datafiles point to the same image
    new_img_locations = {}

    pbar = tqdm(
        enumerate(ds_iterator),
        desc="converting dataset to standard format...",
    )
    final_iter = 0
    for k, (img_path, data) in pbar:
        img_cpt_data = {}
        # get img data
        img_cpt_data.update(data)

        #print(f'img_path: {str(img_path)}')
        if str(img_path) in new_img_locations.keys():
            # if filename is in the dictionary, it already has a new location
            new_img_path = new_img_locations[str(img_path)]["new_img_path"]
            img_cpt_data["image_path"] = new_img_path
            if hash_fn is not None:
                img_cpt_data["metadata"]["image_hash"] = new_img_locations[
                    str(img_path)
                ]["hash"]
        else:
            # if file exists in the old location, it will get moved to a new directory
            #new_img_path = f"images/{save_img_dir}/{img_path}"
            new_img_path = f"{save_img_dir}/{str(img_path).split('/')[-1]}"
            img_cpt_data["image_path"] = new_img_path
            new_img_locations[str(img_path)] = {"new_img_path": new_img_path}
            # original location is saved an later saved to the new directory
            img_path_list.append(img_path)

            # if given, apply hash fn
            if hash_fn is not None:
                try:
                    img = Image.open(img_path).convert("RGB")
                    hash_str = str(hash_fn(img))
                    img_cpt_data["metadata"]["image_hash"] = hash_str
                    # save hash so it does not have to be recomputed
                    new_img_locations[str(img_path)]["hash"] = hash_str
                except (UnidentifiedImageError, FileNotFoundError):
                    print("Warning: corrupted or non-existent Image")

        img_data_list.append(img_cpt_data)

        if k % 10000 - 1 == 0:
            print(f'img_path_list len: {len(img_path_list)}')


        # save images in specified images folder (maximum of dir_size images per folder)
        if (len(img_path_list) % dir_size == 0 and len(img_path_list) > 0):
            print(f"saving {len(img_path_list)} images...")
            os.makedirs(save_img_dir, exist_ok=True)
            save_images(img_path_list, save_img_dir, mode=mode)
            img_path_list = []
            num_img_dirs += 1
            save_img_dir = data_dir / "images" / f"{num_img_dirs}/"

        # save json data in specified image_data folder with consecutive labeling of the json files
        if ((k + 1) % dir_size == 0):
            os.makedirs(save_data_dir, exist_ok=True)
            save_to_jsons(
                img_data_list, save_data_dir, starting_idx=max(k + 1 - dir_size, 0)
            )
            # empty path and data lists and update save directories for next saving step
            img_data_list = []
            save_data_dir = data_dir / "image_data" / f"{int((k+1)/dir_size)}/"
        final_iter = k

    os.makedirs(save_img_dir, exist_ok=True)
    save_images(img_path_list, save_img_dir, mode=mode)
    img_path_list = []
    num_img_dirs += 1
    save_img_dir = data_dir / "images" / f"{num_img_dirs}/"

    os.makedirs(save_data_dir, exist_ok=True)
    save_to_jsons(
                img_data_list, save_data_dir, starting_idx=max(final_iter + 1 - dir_size, 0)
                )
    # empty path and data lists and update save directories for next saving step
    img_data_list = []
    save_data_dir = data_dir / "image_data" / f"{int((final_iter+1)/dir_size)}/"
