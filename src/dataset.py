from PIL import Image
import numpy as np
import config
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ScreenDataset(Dataset):

    def __init__(self, image_paths, labels, train=True):

        self.paths = image_paths
        self.labels = labels

        if train:

            self.transform = A.Compose([

                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),

                A.HorizontalFlip(p=0.5),

                A.RandomBrightnessContrast(
                    brightness_limit=0.08,
                    contrast_limit=0.08,
                    p=0.5
                ),
                A.Rotate(
                    limit=5,
                    p=0.3
                ),

                A.Normalize(
                    mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)
                ),

                ToTensorV2()

            ])

        else:

            self.transform = A.Compose([

                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),

                A.Normalize(
                    mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)
                ),

                ToTensorV2()

            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):

        img=np.array(
            Image.open(self.paths[index]).convert("RGB")
        )

        img=self.transform(image=img)["image"]

        return img,self.labels[index]