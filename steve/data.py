import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.png'):
        self.root = root #/scratch/yy2694/torch_backup/steve-eval/masks_output
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video

from pathlib import Path
import json
class SAYCAMDataset(Dataset):
    def __init__(self, img_dir, json_path, phase, img_size, ep_len=3, img_glob='*.png'):
        self.img_dir = Path("/") / img_dir
        self.img_size = img_size
        self.ep_len = ep_len
        self.phase = phase
        self.json_path = Path(json_path) / str(self.phase + ".json")
        print(json_path)

        with open(self.json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.records = obj['data']

        self.episodes = []
        for idx, entry in enumerate(self.records):
            frame_filenames = entry['frame_filenames']
            frame_buffer = []
            for path in frame_filenames:
                frame_buffer.append(self.img_dir / path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video

from tqdm import *
from PIL import Image, UnidentifiedImageError
class SAMCAMDataset(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.jpg'):
        self.root = root
        self.img_size = img_size
        # self.total_dirs = sorted(glob.glob(root))
        self.total_dirs = []
        self.ep_len = ep_len

        if root == '/saycam_shuffle_5fps/*':
            self.total_dirs = sorted(glob.glob(root))
        elif root == '/data/saycam/5fps/*':
            for dir in sorted(glob.glob(root)):
                if 'S_' in dir:
                    self.total_dirs.append(dir)
        elif root == '/*':
            for dir in glob.glob(root):
                if 'S_' in dir:
                    self.total_dirs.append(dir)

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass
        
        # scan samcam bad iamges
        if root == '/saycam_shuffle_5fps/*':
            bad_pkl = Path("/scratch/yy2694/SlotAttn/steve/saycam_shuffle_5fps_bad_images.pkl")
        elif root == '/data/saycam/5fps/*' or root == '/*':
            bad_pkl = Path("/scratch/yy2694/SlotAttn/steve/5fps_bad_images.pkl")
        if not bad_pkl.exists():
            bad = []
            for dir in tqdm(self.total_dirs):
                for f in glob.glob(os.path.join(dir, img_glob)):
                    if not self._is_valid_iamge(f):
                        bad.append(f)
            # for f in tqdm(glob.glob("/5fps/**/*.jpg", recursive=True)):
            #     if not self._is_valid_iamge(f):
            #         bad.append(f)
            bad_set = set(bad)
            with open(bad_pkl, 'wb') as f:
                import pickle
                pickle.dump(bad_set, f)
        else:
            with open(bad_pkl, 'rb') as f:
                import pickle
                bad_set = pickle.load(f)
        print(f"Found {len(bad_set)} bad images in Samcam dataset.")

        # chunk into episodes
        self.episodes = []
        for dir in tqdm(self.total_dirs):
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                if path in bad_set:
                    continue
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()
    
    def _is_valid_iamge(self, img_path):
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False
        except Exception:
            return False

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video


class GlobVideoDatasetWithMasks(Dataset):
    def __init__(self, root, img_size, num_segs, ep_len=6, img_glob='*_image.png'):
        import pdb; pdb.set_trace()
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        # chunk into episodes
        self.episodes_rgb = []
        self.episodes_mask = []
        for dir in self.total_dirs:
            frame_buffer = []
            mask_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for image_path in image_paths:
                p = Path(image_path)

                frame_buffer.append(p)
                mask_buffer.append([
                    p.parent / f"{p.stem.split('_')[0]}_mask_{n:02}.png" for n in range(num_segs)
                ])

                if len(frame_buffer) == self.ep_len:
                    self.episodes_rgb.append(frame_buffer)
                    self.episodes_mask.append(mask_buffer)
                    frame_buffer = []
                    mask_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes_rgb)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes_rgb[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)

        masks = []
        for mask_locs in self.episodes_mask[idx]:
            frame_masks = []
            for mask_loc in mask_locs:
                image = Image.open(mask_loc).convert('1')
                image = image.resize((self.img_size, self.img_size))
                tensor_image = self.transform(image)
                frame_masks += [tensor_image]
            frame_masks = torch.stack(frame_masks, dim=0)
            masks += [frame_masks]
        masks = torch.stack(masks, dim=0)

        return video, masks
    
class LabeledSDatasetWithMasks(Dataset):
    def __init__(self, root, img_size, num_segs, ep_len=6, img_glob='*_image.png'): #img_8487_mask_08.png
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        # chunk into episodes
        self.episodes_rgb = []
        self.episodes_mask = []
        for dir in self.total_dirs:
            frame_buffer = []
            mask_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for image_path in image_paths:
                p = Path(image_path)

                frame_buffer.append(p)
                mask_buffer.append([
                    p.parent / f"{p.stem.split('_')[0]}_{p.stem.split('_')[1]}_mask_{n:02}.png" for n in range(num_segs)
                ])

                if len(frame_buffer) == self.ep_len:
                    self.episodes_rgb.append(frame_buffer)
                    self.episodes_mask.append(mask_buffer)
                    frame_buffer = []
                    mask_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes_rgb)
    

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes_rgb[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)

        masks = []
        for mask_locs in self.episodes_mask[idx]:
            frame_masks = []
            for mask_loc in mask_locs:
                image = Image.open(mask_loc).convert('1')
                image = image.resize((self.img_size, self.img_size))
                tensor_image = self.transform(image)
                frame_masks += [tensor_image]
            frame_masks = torch.stack(frame_masks, dim=0)
            masks += [frame_masks]
        masks = torch.stack(masks, dim=0)

        return video, masks