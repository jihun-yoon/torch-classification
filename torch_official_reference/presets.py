from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(self,
                 crop_size,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 hflip_prob=0.5,
                 auto_augment_policy=None,
                 random_erase_prob=0.0):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
            trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self,
                 crop_size,
                 resize_size=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 interpolation=InterpolationMode.BILINEAR):

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetTrainCifar10:
    def __init__(
            self,
            crop_size=32,
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
    ):

        self.transforms = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEvalCifar10:
    def __init__(self,
                 mean=(0.5071, 0.4867, 0.4408),
                 std=(0.2675, 0.2565, 0.2761)):

        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.transforms(img)