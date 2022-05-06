import os
import pdb
import glob

from PIL import Image

from torch.utils import data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def make_dataset(img_path):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".jfif")
    list_imgs = []

    for exten in IMG_EXTENSIONS:
        exten_files = glob.glob('{}/*{}'.format(img_path, exten))
        list_imgs = list_imgs + exten_files

    return list_imgs


class demo_loader(data.Dataset):
    """Data loader
    """
    def __init__(
        self,
        dataset_path,
        transforms=None,
    ):
        self.loader = pil_loader
        self.samples = make_dataset(dataset_path)
        self.transform = transforms
        self.imgs = self.samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        im_path = self.samples[index]
        sample = self.loader(im_path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = 0

        return sample, im_path

if __name__ == '__main__':

    data_path = './'
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    custom_dataset = demo_loader(data_path, transformations)

    print('{}'.format(len(custom_dataset)))