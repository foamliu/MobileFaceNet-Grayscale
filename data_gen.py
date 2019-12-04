import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
from config import IMG_DIR, pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data

        self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        filename = os.path.join(IMG_DIR, filename)
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        img = transforms.ToPILImage()(img)
        print(img.size())
        img = self.transformer(img)

        label = sample['label']

        return img, label

    def __len__(self):
        return len(self.samples)
