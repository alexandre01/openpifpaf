import torch.utils.data
import torchvision

from openpifpaf import transforms


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, preprocess=None, image_transform=None):
        self.images = images
        self.preprocess = preprocess
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        pil_image = self.images[index].copy().convert('RGB')

        if self.preprocess is not None:
            pil_image = self.preprocess(pil_image, [], None)[0]

        original_image = torchvision.transforms.functional.to_tensor(pil_image)
        image = self.image_transform(pil_image)

        return 'pilimage{}'.format(index), original_image, image

    def __len__(self):
        return len(self.images)
