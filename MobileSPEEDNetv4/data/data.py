from .aug import CropAndPaste, CropAndPadSafe, DropBlockSafe, AlbumentationAug, RotateImage
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from random import shuffle
from ..utils.EDCoder import Encoder
from ..cfg.config import Config
from threading import Thread
from tqdm import tqdm
from torchvision.transforms import v2

import pickle
import torch
import albumentations as A
import numpy as np
import cv2 as cv
import lightning as L


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class ImageReader(Thread):
    def __init__(self, img_name: list, config: dict, image_dir: Path):
        Thread.__init__(self)
        self.config: dict = config
        self.image_dir: Path = image_dir
        self.image_name: list = img_name
        self.img_dict: dict = {}
    
    def run(self):
        for img_name in tqdm(self.image_name):
            image = cv.imread(str(self.image_dir / img_name), cv.IMREAD_GRAYSCALE)
            self.img_dict[img_name] = image
    
    def get_result(self) -> dict:
        return self.img_dict


class Speed(Dataset):
    init: bool = False
    
    def __init__(self, config: Config, mode: str = "train"):
        if not Speed.init:
            Speed.config = config
            Speed.data_dir = Path(config.data_dir)
            Speed.image_dir = Speed.data_dir / "images/train"
            Speed.label_file = Speed.data_dir / "train_label.pkl"
            
            # 标签
            with open(Speed.label_file, "rb") as f:
                Speed.train_val_labels = pickle.load(f)
            
            # 采样列表
            Speed.image_list = list(Speed.train_val_labels.keys())[:500] if config.debug else list(Speed.train_val_labels.keys())
            shuffle(Speed.image_list)
            total_num = len(Speed.image_list)
            Speed.train_len = int(total_num * config.split[0])
            Speed.val_len = total_num - Speed.train_len
            Speed.train_list = Speed.image_list[:Speed.train_len]
            Speed.val_list = Speed.image_list[Speed.train_len:]
            
            # 缓存图片
            if Speed.config.ram:
                Speed.image_dict = {}
                Speed.read_image()
            
            # resize
            Speed.Resize = A.Compose([A.Resize(height=Speed.config.imgsz[0], width=Speed.config.imgsz[1], p=1.0, interpolation=cv.INTER_LINEAR)],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
            
            # 姿态编码器
            Speed.encoder = Encoder(Speed.config.stride, Speed.config.alpha, Speed.config.neighbor, device=Speed.config.device)
            
            # 转换为Tensor
            Speed.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            
            Speed.init = True
        
        self.mode = mode
    
    def __len__(self):
        return self.train_len if self.mode == "train" else self.val_len

    def __getitem__(self, index):
        image_name = Speed.train_list[index] if self.mode == "train" else Speed.val_list[index]
        if Speed.config.ram:
            image = Speed.image_dict[image_name]
        else:
            image = cv.imread(str(Speed.image_dir / image_name), cv.IMREAD_GRAYSCALE)
        
        pos, ori, bbox = Speed.train_val_labels[image_name]["pos"], Speed.train_val_labels[image_name]["ori"], Speed.train_val_labels[image_name]["bbox"]
        
        if Speed.config.resize_first:
            transformed = self.Resize(image=image, bboxes=bbox.reshape(1, 4), category_ids=[1])
            image, bbox = transformed["image"], transformed["bboxes"].reshape(4).astype(np.int32)
        
        if self.mode == "train":
            image = CropAndPaste(image, bbox, p=Speed.config.CropAndPaste_p)
            image = CropAndPadSafe(image, bbox, p=Speed.config.CropAndPadSafe_p)
            image = DropBlockSafe(image, bbox, drop_n=Speed.config.drop_n, p=Speed.config.DropBlockSafe_p)
            image, pos, ori, bbox = RotateImage(image, pos, ori, bbox, max_rotate_angle=Speed.config.max_rotate_angle, limit_num=Speed.config.limit_num, p=Speed.config.RotateImage_p)
            image = AlbumentationAug(image, p=Speed.config.AlbumentationAug_p)
        
        if not Speed.config.resize_first:
            transformed = self.Resize(image=image, bboxes=bbox.reshape(1, 4), category_ids=[1])
            image, bbox = transformed["image"], np.array(transformed["bboxes"][0]).reshape(4).astype(np.int32)
        
        image = Speed.transform(image)
        yaw_encode, pitch_encode, roll_encode = Speed.encoder.encode_ori(ori)
        
        label = {
            "filename": image_name,
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "bbox": bbox.astype(np.int32),
            "yaw_encode": yaw_encode.astype(np.float32),
            "pitch_encode": pitch_encode.astype(np.float32),
            "roll_encode": roll_encode.astype(np.float32)
        }
        
        return image, label
        
    @staticmethod
    def divide_data(lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    @staticmethod
    def read_img(thread_num: int = 12):
        # 将采样列表中的图片读入内存
        img_divided = Speed.divide_data(Speed.img_name, thread_num)
        thread_list = []
        for sub_img_name in img_divided:
            thread_list.append(ImageReader(sub_img_name, Speed.config, Speed.image_dir))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
        for thread in thread_list:
            Speed.img_dict.update(thread.get_result())
        
        
class SpeedDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Speed(self.config, "train")
            self.val_dataset = Speed(self.config, "val")
        elif stage == "validate":
            self.val_dataset = Speed("val")
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def train_dataloader(self):
        if self.config.debug:
            loader = DataLoader(
                self.train_dataset,
                batch_size=Speed.config.batch_size,
                shuffle=True,
                num_workers=Speed.config.num_workers,
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device="cuda"
            )
        else:
            loader = MultiEpochsDataLoader(
                self.train_dataset,
                batch_size=Speed.config.batch_size,
                shuffle=True,
                num_workers=Speed.config.num_workers,
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device="cuda"
            )
        return loader

    def val_dataloader(self):
        if self.config.debug:
            loader = DataLoader(
                self.val_dataset,
                batch_size=Speed.config.batch_size,
                shuffle=False,
                num_workers=Speed.config.num_workers,
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device="cuda"
            )
        else:
            loader = MultiEpochsDataLoader(
                self.val_dataset,
                batch_size=Speed.config.batch_size,
                shuffle=False,
                num_workers=Speed.config.num_workers,
                persistent_workers=True,
                pin_memory=True,
                pin_memory_device="cuda"
            )
        return loader