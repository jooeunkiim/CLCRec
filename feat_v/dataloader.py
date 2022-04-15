import os
import random
import numpy as np
from tqdm import tqdm

import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, data_dir='/data/private/videos', transform=None):
        # lst = os.listdir(data_dir)
        lst = np.load("/data/private/CLCRec/Data/movie/item_list.npy")
        self.dir = data_dir
        self.images = []
        for movie in lst:
            files = os.listdir(os.path.join(data_dir, str(movie)))
            self.images.extend(random.sample([f for f in files if f.endswith(".jpg")], 1))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]
        
    def custom_collate_fn(self, data):
        inputImages = []
        outputVectors = []

        for sample in data:
            movielens_id = int(sample.split("-")[0])
            img = cv2.imread(os.path.join(self.dir, str(movielens_id), sample))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            inputImages.append(img)
            outputVectors.append(movielens_id)

        data = {'movielens_id': outputVectors, 'input': inputImages}

        if self.transform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        filename, input = data['movielens_id'], data['input']
        width, height = input[0].shape[:2]
        
        input_tensor = torch.empty(len(input), 3, width, height) # if RGB
        # input_tensor = torch.empty(len(input), width, height) # if GRAY
        
        for i in range(len(input)):
            input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
            input_tensor[i] = torch.from_numpy(input[i]) 
        
        # input_tensor = torch.unsqueeze(input_tensor, 1) # if GRAY scale

        return {'movielens_id': filename, 'input': input_tensor}

class Normalize(object):
    def __call__(self, data):
        filename, input = data['movielens_id'], data['input']
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = normalize(input)

        return {'movielens_id': filename, 'input': input}
