from pathlib import Path
# import json
# import pickle as pkl
# import time

# import cv2
# import lmdb
# import numpy as np
# import torch as pt
# import torch.utils.data as ptud

from .utils import (
    rgb_segment_to_index_segment,
    index_segment_to_bbox,
    draw_segmentation_np,
)

# class SAYCAM(ptud.Dataset):
#     def __init__(
#         self,
#         data_file,
#         split,
#         filelist,
#         exts:tuple = (".jpg", ".jpeg", ".png"),
#         recursive: bool = True,
#         transform=lambda **_: _,
#         max_spare=4,
#         base_dir: Path = None,
#     ):
#         if base_dir:
#             data_file = base_dir / data_file
#         print("data_file:", data_file)

#         self.env = lmdb.open(
#             str(data_file),
#             subdir=False,
#             readonly=True,
#             readahead=False,
#             meminit=False,
#             max_spare_txns=max_spare,
#             lock=False,
#         )
#         with self.env.begin(write=False) as txn:
#             self.idxs = pkl.loads(txn.get(b"__keys__"))
#         self.instance = instance
#         self.extra_keys = extra_keys
#         self.transform = transform

#     def __getitem__(self, index, compact=True):
#         """
#         - image: shape=(c=3,h,w), uint8
#         - bbox: shape=(n,c=4), float32
#         - segment: shape=(h,w), uint8
#         - clazz: shape=(n,), uint8
#         - isthing: shape=(n,), bool
#         """
#         # load sample pack
#         with self.env.begin(write=False) as txn:
#             sample0 = pkl.loads(txn.get(self.idxs[index]))
#         sample1 = {}

#         # load image and segment
#         image0 = cv2.cvtColor(
#             cv2.imdecode(  # cvtColor will unify images to 3 channels safely
#                 np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
#             ),
#             cv2.COLOR_BGR2RGB,
#         )
#         image = pt.from_numpy(image0).permute(2, 0, 1)
#         sample1["image"] = image

#         def bytes2uint8(x):
#             if isinstance(x, bytes):
#                 x = np.frombuffer(x, dtype=np.uint8)
#             return x
        
#         if "segment" in self.extra_keys:
#             # print("keys:", sample0.keys())
#             # print(sample0["segment"])
#             segment = pt.from_numpy(
#                 cv2.imdecode(bytes2uint8(sample0["segment"]), cv2.IMREAD_GRAYSCALE)
#             )
#             sample1["segment"] = segment

#             # load bbox and clazz for set prediction
#             if "bbox" in self.extra_keys:
#                 bbox = pt.from_numpy(sample0["bbox"])
#                 sample1["bbox"] = bbox
#             if "clazz" in self.extra_keys:
#                 clazz = pt.from_numpy(sample0["clazz"])
#                 sample1["clazz"] = clazz

#         # conduct transformation
#         sample2 = self.transform(**sample1)

#         if "segment" in self.extra_keys:
#             # merge stuff into the background, idx=1
#             # 0: bg or no-annotat; >0: fg (things & stuff for panoptic, things only for instance)
#             isthing = pt.from_numpy(sample0["isthing"])
#             segment = sample2["segment"]
#             if self.instance:  # set all stuff indexes to 0 as the background
#                 stuff_idxs = (~isthing).nonzero() + 1
#                 segment[pt.isin(segment, stuff_idxs)] = 0
#                 sample2["segment"] = segment

#             # remove invalid bbox and clazz
#             isvalid = pt.zeros_like(isthing)
#             sidxs0 = segment.unique().tolist()
#             sidxs0 = list(set(sidxs0) - set(range(MSCOCO.OFFSET)))
#             sidxs0.sort()
#             isvalid[pt.from_numpy(np.array(sidxs0, "int64")) - MSCOCO.OFFSET] = 1
#             if "bbox" in self.extra_keys:
#                 sample2["bbox"] = sample2["bbox"][isvalid]
#             if "clazz" in self.extra_keys:
#                 sample2["clazz"] = sample2["clazz"][isvalid]

#             # compact segment idxs to be continuous
#             if compact:
#                 # seg2 = sample2["segment"]  # not work for segment with (sporadic) offset
#                 # sidxs, seg2 = seg2.unique(return_inverse=True)
#                 # seg2 = seg2.reshape(seg2.shape).byte()
#                 # delta = len(set(range(MSCOCO.OFFSET)) - set(sidxs.numpy().tolist()))
#                 # seg2 += delta
#                 segment = sample2["segment"]
#                 sidxs = segment.unique().tolist()
#                 cnt = MSCOCO.OFFSET
#                 for sidx in sidxs:
#                     if sidx in range(MSCOCO.OFFSET):  # 0: bg or no-annotat
#                         continue
#                     segment[segment == sidx] = cnt
#                     cnt += 1
#                 # assert (seg2 == segment).all()
#                 sample2["segment"] = segment

#         return sample2

#     def __len__(self):
#         return len(self.idxs)

import os
import glob
import torch, torchvision

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import json
from tqdm import *

from torchvision import transforms
from torchvision.transforms import InterpolationMode

class SAYCAM(Dataset):
    def __init__(self, 
                 data_file, 
                 split, 
                 transform=lambda **_: _,
                 base_dir: Path = None,
                 ):
        self.root = data_file
        # if base_dir is not None:
        #     self.root = base_dir
        
        self.total_paths = sorted(glob.glob(os.path.join(self.root+'S*/*.jpg')))                

        if split == 'train':
            self.total_paths = self.total_paths[:int(len(self.total_paths) * 0.7)]
        elif split == 'val':
            self.total_paths = self.total_paths[int(len(self.total_paths) * 0.7):int(len(self.total_paths) * 0.85)]
        elif split == 'test':
            self.total_paths = self.total_paths[int(len(self.total_paths) * 0.85):]
        else:
            pass
        # if split == 'train':
        #     self.total_paths = self.total_paths[:100]
        # elif split == 'val':
        #     self.total_paths = self.total_paths[100:150]
        # elif split == 'test':
        #     self.total_paths = self.total_paths[150:200]
        # else:
        #     pass
        
        bad_pkl = Path("/scratch/wz3008/SlotAttn/steve/5fps_bad_images.pkl")
        with open(bad_pkl, 'rb') as f:
            import pickle
            bad_set = pickle.load(f)
        
        self.total_imgs = []
        for path in tqdm(self.total_paths):
            if path not in bad_set:
                self.total_imgs.append(path)
            
        
        IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
        IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
        resolut0 = [256, 256]
        if split == 'train':
            self.transform = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    size=resolut0, 
                    scale=(0.75, 1.0),
                    ratio=(1.0,1.0),
                    interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif split == 'val':
            self.transform = torchvision.transforms.Compose([
                transforms.Resize(resolut0, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolut0),
                transforms.ToTensor(),                
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # video = []
        # for img_loc in self.episodes[idx]:
        #     image = Image.open(img_loc).convert("RGB")
        #     image = image.resize((self.img_size, self.img_size))
        #     tensor_image = self.transform(image)
        #     video += [tensor_image]
        # video = torch.stack(video, dim=0)
        ret = {}
        img = Image.open(self.total_imgs[idx]).convert("RGB")
        # img = torch.from_numpy(np.array(img)).permute(2,0,1).contiguous()
        ret['image'] = self.transform(img)
        return ret