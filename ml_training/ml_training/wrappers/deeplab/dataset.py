from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


class SegmentationDataset(Dataset):
    def __init__(
            self,
            image_paths,
            mask_paths,
            num_classes,
            channels=3,
            preprocess_image=None,
            preprocess_mask=None,
            transform=None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.num_classes = num_classes
        self.preprocess_image = preprocess_image
        self.preprocess_mask = preprocess_mask
        self.transform = transform

        self.colors = self.__get_colors()

    def show_image(self, idx):
        image = self.image_paths[idx]
        mask = self.mask_paths[idx]

        image = cv2.imread(image)
        mask = cv2.imread(mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transforms = self.transform(image=image, mask=mask)
            image, mask = transforms['image'], transforms['mask']

        res = np.concatenate((image, mask), axis=1)

        plt.imshow(res)
        plt.show()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        mask = self.mask_paths[idx]

        image = cv2.imread(image)
        mask = cv2.imread(mask)

        if self.channels == 1 and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform:
            transforms = self.transform.img_mask(image=image, mask=mask)
            image, mask = transforms['image'], transforms['mask']
            image = self.transform.img(image=image)['image']

        image = self.preprocess_image(image=image)['image']

        mask = self.__convert_mask_to_multichannel(mask, self.colors)
        mask = self.preprocess_mask(image=mask)['image']
        mask = torch.where(mask > 0, torch.tensor(1), torch.tensor(0))
        mask = mask.to(torch.float32)

        return image, mask

    def __get_colors(self):
        colors = []
        i = 0
        while len(colors) < self.num_classes:
            mask = cv2.imread(self.mask_paths[i])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            colors_ = np.unique(mask)

            for color in colors_:
                if color not in colors:
                    colors.append(color)

            i += 1
            if i == len(self.mask_paths):
                break

        colors = np.sort(colors)

        return colors

    def __convert_mask_to_multichannel(self, mask, colors):
        multichannel_mask = np.zeros((mask.shape[0], mask.shape[1], len(colors)), dtype=np.float32)
        for i, color in enumerate(colors):
            multichannel_mask[:, :, i] = (mask == color).astype(np.float32)

        return multichannel_mask
