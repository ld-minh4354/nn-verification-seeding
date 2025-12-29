import os, sys
import shutil

from PIL import Image

import numpy as np



class DataProcessingJPL:
    def __init__(self):
        self.add_project_folder_to_pythonpath()

        self.jpl_root = os.path.join("raw_datasets", "JPL")
        self.jpl_processed_root = os.path.join("raw_datasets", "JPL_processed")

        train_folder_background_path = os.path.join(self.jpl_processed_root, "train", "background")
        train_folder_frost_path      = os.path.join(self.jpl_processed_root, "train", "frost")
        test_folder_background_path  = os.path.join(self.jpl_processed_root, "test", "background")
        test_folder_frost_path       = os.path.join(self.jpl_processed_root, "test", "frost")

        self.path_list = [train_folder_background_path, train_folder_frost_path,
                     test_folder_background_path, test_folder_frost_path]


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def main(self):
        # self.get_data_split()
        # self.initialize_data_folders()
        # self.process_all_images()
        self.count_data_size()
        # self.calculate_stats()
        return


    def get_data_split(self):
        train_id_path = os.path.join(self.jpl_root, "data_split", "train_source_images.txt")
        test_id_path  = os.path.join(self.jpl_root, "data_split", "test_source_images.txt")

        with open(train_id_path, "r") as f:
            content = f.read()
            self.train_id = content.split("\n")
        
        with open(test_id_path, "r") as f:
            content = f.read()
            self.test_id = content.split("\n")


    def initialize_data_folders(self):
        for path in self.path_list:
            shutil.rmtree(path)
            os.makedirs(path)

    
    def process_all_images(self):
        for subfolder in os.listdir(os.path.join(self.jpl_root, "data")):
            image_id = "_".join(subfolder.split("_")[:3])
            
            if image_id in self.train_id:
                self.process_image(subfolder, "train")
            if image_id in self.test_id:
                self.process_image(subfolder, "test")


    def process_image(self, subfolder, split):
        current_dir = os.path.join(self.jpl_root, "data", subfolder, "tiles")

        image_class = None
        for subfolder in os.listdir(current_dir):
            image_class = subfolder
        
        current_dir = os.path.join(current_dir, image_class)
        target_dir = os.path.join(self.jpl_processed_root, split, image_class)

        for filename in os.listdir(current_dir):
            file_path =   os.path.join(current_dir, filename)
            target_path = os.path.join(target_dir, filename)

            if not os.path.isfile(file_path):
                raise Exception(f"Abnormal folder path: {file_path}")
            
            with Image.open(file_path) as img:
                img = img.convert("L")
                img_resized = img.resize((28, 28), resample=Image.BICUBIC)
                img_resized.save(target_path)


    def count_data_size(self):
        for path in self.path_list:
            count = sum(
                1 for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            )
            print(f"{path}: {count} instances")

    
    def calculate_stats(self):
        train_dirs = [
            os.path.join(self.jpl_processed_root, "train", "background"),
            os.path.join(self.jpl_processed_root, "train", "frost")
        ]

        sum_pixels = 0.0
        sum_squared = 0.0
        num_pixels = 0

        for dir in train_dirs:
            for filename in os.listdir(dir):
                if not filename.lower().endswith(".png"):
                    continue

                img_path = os.path.join(dir, filename)
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    arr = np.asarray(img, dtype=np.float64) / 255.0

                sum_pixels += arr.sum()
                sum_squared += np.square(arr).sum()
                num_pixels += arr.size
        
        mean = sum_pixels / num_pixels
        std = np.sqrt(sum_squared / num_pixels - mean ** 2)

        print(f"Mean: {mean}")
        print(f"Std: {std}")



if __name__ == "__main__":
    dp = DataProcessingJPL()
    dp.main()
