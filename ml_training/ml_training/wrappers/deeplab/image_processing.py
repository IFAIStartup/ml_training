import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import easydict


def preproc_image(img_size, channels=3):
    if channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    elif channels == 1:
        mean = [0.5]
        std = [0.5]

    else:
        raise ValueError('channels must be 1 or 3')

    transform = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=0
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )

    return transform


def preproc_mask(img_size):
    transform = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
            ),
            ToTensorV2(),
        ]
    )

    return transform


def get_augmentations():
    args = easydict.EasyDict({
        "img_mask": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.3,
                    rotate_limit=30,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    mask_value=0,
                    p=0.33,
                ),
                A.Perspective(p=0.1),
            ]
        ),
        "img": A.Compose(
            [
                A.RandomBrightnessContrast(p=0.25),
                A.RandomGamma(p=0.25),

            ]
        ),
    })

    return args


def preproc_image_roads(img_size, channels=3):
    if channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    elif channels == 1:
        mean = [0.5]
        std = [0.5]

    else:
        raise ValueError('channels must be 1 or 3')

    transform = A.Compose(
        [
            A.RandomCrop(height=img_size * 2, width=img_size * 2, p=1.0),
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=0
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )

    return transform
