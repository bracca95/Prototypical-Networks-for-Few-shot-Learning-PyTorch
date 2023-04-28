import os
import torch

from src.tools import Utils, Logger

from PIL import Image
from glob import glob
from typing import Optional, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DefectViews(Dataset):

    label_to_idx = {
        "bubble": 0, 
        "point": 1,
        "break": 2,
        "dirt": 3,
        "mark": 4,
        "scratch": 5
    }

    idx_to_label = Utils.invert_dict(label_to_idx)

    def __init__(self, dataset_path: str, crop_size: int, img_size: Optional[int] = 105, filt: Optional[List[str]] = None):
        self.dataset_path: str = dataset_path
        self.filt: Optional[List[str]] = filt
        self.image_list: Optional[List[str]] = self.get_image_list()
        self.label_list: Optional[List[int]] = self.get_label_list()

        self.crop_size = crop_size
        self.img_size = img_size
        self.in_dim = self.img_size if self.img_size is not None else self.crop_size
        self.out_dim = len(self.label_to_idx)

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def get_image_list(self) -> List[str]:
        image_list = [f for f in glob(os.path.join(self.dataset_path, "*.png"))]
        
        if self.filt is not None:
            filenames = list(map(lambda x: os.path.basename(x), image_list))
            image_list = list(filter(lambda x: Utils.check_string(x.rsplit("_")[0], self.filt, True, False), filenames))
            image_list = list(map(lambda x: os.path.join(self.dataset_path, x), image_list))
        
        if not all(map(lambda x: x.endswith(".png"), image_list)) or image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")

        return image_list

    def get_label_list(self) -> List[int]:
        if self.image_list is None:
            self.get_image_list()

        filenames = list(map(lambda x: os.path.basename(x), self.image_list))
        label_list = list(map(lambda x: x.rsplit("_")[0], filenames))

        Logger.instance().debug(f"Labels used: {set(label_list)}")
        Logger.instance().debug(f"Number of images per class: { {i: label_list.count(i) for i in set(label_list)} }")
        
        return [self.label_to_idx[defect] for defect in label_list]

    def load_image(self, path: str) -> torch.Tensor:
        # about augmentation https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
        img_pil = Image.open(path).convert("L")

        # crop
        # TODO: cropping by the shortest side may result in cropped breaks/scratches (rectangular shaped). Instead,
        #       reshaping may alter image proportions.
        if img_pil.size[0] * img_pil.size[1] < self.crop_size * self.crop_size:
            m = min(img_pil.size)
            centercrop = transforms.Compose([transforms.CenterCrop((m, m))])
            resize = transforms.Compose([transforms.Resize((self.crop_size, self.crop_size))])
            
            img_pil = centercrop(img_pil)
            img_pil = resize(img_pil)

            Logger.instance().debug(f"image size for {os.path.basename(path)} is less than required. Upscaling.")
        elif any(map(lambda x: x in os.path.basename(path), ["mark", "scratch", "break"])):
            resize = transforms.Compose([transforms.Resize((self.crop_size, self.crop_size))])
            img_pil = resize(img_pil)
        else:
            centercrop = transforms.Compose([transforms.CenterCrop((self.crop_size, self.crop_size))])
            img_pil = centercrop(img_pil)
        
        # resize (if required)
        if self.img_size is not None:
            resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
            img_pil = resize(img_pil)

        # rescale [0-255](int) to [0-1](float)
        totensor = transforms.Compose([transforms.ToTensor()])
        img = totensor(img_pil)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize = transforms.Normalize(self.mean, self.std)
            img = normalize(img)

        return img # type: ignore

    @staticmethod
    def compute_mean_std(dataset: Dataset):
        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=32)

        mean = 0.0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dataloader.dataset)

        var = 0.0
        pixel_count = 0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)

        if any(map(lambda x: torch.isnan(x), mean)) or any(map(lambda x: torch.isnan(x), std)):
            raise ValueError("mean or std are none")

        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]
        curr_label_batch = self.label_list[index]
        
        # to check: if I use bubble and breaks (0, 2), label prediction via softmax is (0, 1), so wrong?
        return self.load_image(curr_img_batch), curr_label_batch

    def __len__(self):
        return len(self.image_list) # type: ignore
    

class NoBreaks(DefectViews):

    label_to_idx = {
        "bubble": 0, 
        "point": 1,
        "dirt": 2,
        "mark": 3,
        "scratch": 4
    }

    idx_to_label = Utils.invert_dict(label_to_idx)

    def __init__(self, dataset_path: str, crop_size: int, img_size: Optional[int] = 105, filt: Optional[List[str]] = ["bubble", "point", "dirt", "mark", "scratch"]):
        super().__init__(dataset_path, crop_size, img_size, filt)