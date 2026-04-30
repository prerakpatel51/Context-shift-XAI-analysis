from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(split: str, image_size: int = 224, augmentation: str = "strong"):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if split == "train" and augmentation == "strong":
        aug = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        ]
        try:
            aug.append(transforms.RandAugment())
        except Exception:
            pass
        aug += [transforms.ToTensor(), normalize, transforms.RandomErasing(p=0.25)]
        return transforms.Compose(aug)
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
