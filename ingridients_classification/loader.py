import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json

total_number_ingridients = 556 # valid ingridients form 1 to 555 + empty = 0


def parse_ingridients(data:list):

    # dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)

    dish_code = data[0]
    ingrs = np.zeros((total_number_ingridients,))
    data = data[6:] #removing stuff about dish
    cells_data_per_ingr = 7
    num_ingrs = len(data) // cells_data_per_ingr
    for i in range(num_ingrs):
        ingr_id_name = data[i * cells_data_per_ingr] # ingr_0000000xxx
        id = int(ingr_id_name[-5:])
        ingrs[id] = 1
    return dish_code, ingrs

    


def readCsvData(filepath):
    if not os.path.exists(filepath):
        raise Exception("File %s not found" % os.path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        filelines = f_in.readlines()
        for line in filelines:
            data_values = line.strip().split(",")
            dish_id, ingrs = parse_ingridients(data_values)
            parsed_data[dish_id] = ingrs
    return parsed_data



class FoodDataset(Dataset):
    def __init__(self, dataset_path="./nutrition5k_dataset_nosides/", mode='train', augument=3, cache=True):
        assert mode in ['train', 'test', 'small_train']
        if mode != 'train':
            augument = 1

        dataset_path = os.path.abspath(dataset_path)
        self.images_path = os.path.join(dataset_path, "imagery/realsense_overhead")
        self.cache = cache

        self.rgb_filename = "rgb.png"


        #mappings from old to new categories, where classes with no data are removed.
        f = open('mappings.json')
        mappings = json.load(f) 
        self.old_to_new_mapping = mappings['old_to_new']
        self.new_to_old_mapping = mappings['new_to_old']

        
        prefix = 'small_' if mode == 'small_train' else ''
        train_path = os.path.join(dataset_path,f"dish_ids/splits/{prefix}rgb_train_ids.txt")
        test_path = os.path.join(dataset_path,f"dish_ids/splits/{prefix}rgb_test_ids.txt")

        ids_path = test_path if mode == 'test' else train_path

        labels_path = os.path.join(dataset_path, "metadata")
        labels_first_file = os.path.join(labels_path,"dish_metadata_cafe1.csv")
        labels_second_file = os.path.join(labels_path,"dish_metadata_cafe2.csv")

        labels = readCsvData(labels_first_file)
        labels.update(readCsvData(labels_second_file))



        self.transform = transforms.Compose([
            transforms.ToTensor(),               # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image,ì
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
            transforms.RandomRotation(degrees=15),  # Randomly rotate the image
            transforms.RandomResizedCrop(224),   # Randomly crop the image and resize it to 224x224
        ])


        self.data = []
        self.labels = []

        with open(ids_path, "r") as f_in:
            filelines = f_in.readlines()
            for line in tqdm(filelines, desc="pre-processing"):
                line = line.strip()
                for i in range(augument):
                    if cache:
                        rgb_image = self.load_rgb(os.path.join(self.images_path,line,self.rgb_filename), self.transform)

                        if (rgb_image is None):
                            #print(f"Problems with data in {line}.. skipping.")
                            continue
                        data = rgb_image
                    else:
                        data = os.path.join(self.images_path, line)
                        rgb_image = self.load_rgb(os.path.join(data, self.rgb_filename))
                        if (rgb_image is None):
                            #print(f"Problems with data in {os.path.join(data, self.rgb_filename)}.. skipping.")
                            continue

                    self.data.append(data)


                    label = np.zeros(len(self.new_to_old_mapping))
                    old_label = labels[line]
                    for i in range(len(old_label)):
                        if not self.old_to_new_mapping[i] is None:
                            label[self.old_to_new_mapping[i]] = old_label[i]

                    label = torch.from_numpy(label)
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
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache:
            sample = self.data[index]
        else:
            
            sample = self.load_rgb(os.path.join(self.data[index], self.rgb_filename), self.transform)
            if sample is None:
                print("ERROR in ", os.path.join(self.data[index], self.rgb_filename))

        label = self.labels[index]
        return sample, label



def main():
    dataset = FoodDataset(mode='train')
    print("Dataset items:", len(dataset))
    return

if __name__ == '__main__':
   main()