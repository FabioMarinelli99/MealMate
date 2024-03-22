import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

total_number_ingridients = 556 # valid ingridients form 1 to 555 + empty = 0

def readCsvData(filepath):
    if not os.path.exists(filepath):
        raise Exception("File %s not found" % os.path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        filelines = f_in.readlines()
        for line in filelines:
            data_values = line.strip().split(",")
            parsed_data[data_values[0]] = np.array([float(i) for i in data_values[2:6]])
    return parsed_data


class FoodDataset(Dataset):
    def __init__(self, dataset_path="./nutrition5k_dataset_nosides/", mode='train', only_rgb=False, augument=3, cache=True):
        assert mode in ['train', 'test', 'small_train']
        if mode != 'train':
            augument = 1

        dataset_path = os.path.abspath(dataset_path)
        self.images_path = os.path.join(dataset_path, "imagery/realsense_overhead")
        self.cache = cache
        self.only_rgb = only_rgb

        self.rgb_filename = "rgb.png"

        if not only_rgb:
            self.depth_filename = "depth_raw.png"

        # TODO: includere calcolo di media e varianza
        avg_stats_file = "./dataset4_avg.npy"
        var_stats_file = "./dataset4_var.npy"

        assert os.path.isfile(avg_stats_file) and os.path.isfile(var_stats_file)
        self.avg = torch.from_numpy(np.load(avg_stats_file))
        self.var = torch.from_numpy(np.load(var_stats_file))
        
        key = 'rgb' if only_rgb else 'depth'
        prefix = 'small_' if mode == 'small_train' else ''
        train_path = os.path.join(dataset_path,f"dish_ids/splits/{prefix}{key}_train_ids.txt")
        test_path = os.path.join(dataset_path,f"dish_ids/splits/{prefix}{key}_test_ids.txt")

        ids_path = test_path if mode == 'test' else train_path

        labels_path = os.path.join(dataset_path, "metadata")
        labels_first_file = os.path.join(labels_path,"dish_metadata_cafe1.csv")
        labels_second_file = os.path.join(labels_path,"dish_metadata_cafe2.csv")

        labels = readCsvData(labels_first_file)
        labels.update(readCsvData(labels_second_file))


        transform_all = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
            transforms.RandomRotation(degrees=15),  # Randomly rotate the image
            transforms.RandomResizedCrop(224),   # Randomly crop the image and resize it to 224x224
            
        ])

        transform_rgb = transforms.Compose([
            transforms.ToTensor(),               # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        self.transform_rgb = transform_rgb
        self.transform_depth = transforms.ToTensor()
        self.transform_common = transform_all

        self.data = []
        self.labels = []

        with open(ids_path, "r") as f_in:
            filelines = f_in.readlines()
            for line in tqdm(filelines, desc="pre-processing.."):
                line = line.strip()
                for i in range(augument):
                    if cache:
                        rgb_image = self.load_rgb(os.path.join(self.images_path,line,self.rgb_filename), self.transform_rgb)
                        if not only_rgb:
                            depth_image = self.load_depth(os.path.join(self.images_path,line,self.depth_filename), self.transform_depth)
                        
                        if (rgb_image is None) or (depth_image is None):
                            print(f"Problems with data in {line}.. skipping.")
                            continue

                        if not only_rgb:
                            data = torch.cat((rgb_image, depth_image), dim=0)
                        else:
                            data = rgb_image
                        data = self.transform_common(data)
                    else:
                        data = os.path.join(self.images_path, line)
                        rgb_image = self.load_rgb(os.path.join(data, self.rgb_filename))
                        depth_image = self.load_depth(os.path.join(data, self.depth_filename))

                        if (rgb_image is None) or (depth_image is None):
                            print(f"Problems with data in {line}.. skipping.")
                            continue

                    self.data.append(data)
                    label = torch.from_numpy(labels[line])

                    label = (label - self.avg) / self.var
                    self.labels.append(label)



    def load_rgb(self, file_name:str, transform=None):
        try:
            if not os.path.isfile(file_name) or os.path.getsize(file_name) == 0:
                return None

            image = Image.open(file_name).convert('RGB')
            if not transform is None:
                image = transform(image)
            return image
        except:
            return None
        
    def load_depth(self, file_name:str, transform=None):
        try:

            if not os.path.isfile(file_name) or os.path.getsize(file_name) == 0:
                return None
            
            # Open the image using PIL
            depth_image = Image.open(file_name)
            
            # Convert PIL image to numpy array
            depth_array = np.array(depth_image)
            
            # Scale the depth values to the desired range (0 to 0.4 meters)
            image = depth_array.astype(np.float32) / 10000.0 * 0.4

            if not transform is None:
                image = transform(image)
            return image
        except:
            return None

    def split_channels(self, image):
        rgb, deep = image[:3, :, :], image[3:, :, :]
        deep = torch.cat((deep, deep, deep), dim=0)
        return rgb, deep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.cache:
            sample = self.data[index]
        else:
            
            rgb_image = self.load_rgb(os.path.join(self.data[index], self.rgb_filename), self.transform_rgb)
            if rgb_image is None:
                print("ERROR in ", os.path.join(self.data[index], self.rgb_filename))
            if not self.only_rgb:
                depth_image = self.load_depth(os.path.join(self.data[index], self.depth_filename), self.transform_depth)
                if depth_image is None:
                    print("ERROR in ", os.path.join(self.data[index], self.depth_filename))

            if not self.only_rgb:
                sample = torch.cat((rgb_image, depth_image), dim=0)
            else:
                sample = rgb_image
            
            sample = self.transform_common(sample)
            

        rgb, deep = self.split_channels(sample)
        label = self.labels[index]
        return rgb, deep, label



def main():
    dataset = FoodDataset(mode='train', only_rgb=False)
    print("Dataset items:", len(dataset))

    return

if __name__ == '__main__':
   main()