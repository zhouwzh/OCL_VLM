# 对于每个视频中的每一帧，这一帧有N个物体，那么存储n个组合，每个组合包括：
# （1）原本图片的存储位置
# （2）movi-a本身提供的，这个物体的mask的存储位置
# （3）这个物体对应的selected slot feature 和slot mask（由一个steve model得到的tensor向量）
# （4）这个物体对应的caption
import argparse
import sys
import os
import json
from pathlib import Path
from tqdm import *

import numpy as np
import torch
import tensorflow_datasets as tfds
from torchvision import transforms
import torchvision.utils as vutils
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt

def get_object_caption(example, ds_info, inst_idx: int)->str:
    # structure of example:
    # instances:
    # metadata:
    # object_coordinates:
    # segmentations:
    # videos:


    inst = example["instances"]
    labels = []

    if "size_label" in inst:
        size_idx = inst["size_label"][inst_idx]
        labels.append(ds_info.features["instances"]["size_label"].names[int(size_idx)])

    if "color_label" in inst:
        color_idx = inst["color_label"][inst_idx]
        labels.append(ds_info.features["instances"]["color_label"].names[int(color_idx)])

    if "material_label" in inst:
        material_idx = inst["material_label"][inst_idx]
        labels.append(ds_info.features["instances"]["material_label"].names[int(material_idx)])

    if "shape_label" in inst:
        shape_idx = inst["shape_label"][inst_idx]
        labels.append(ds_info.features["instances"]["shape_label"].names[int(shape_idx)])

    return " ".join(labels)

def iou_bool(a:np.ndarray, b:np.ndarray, eps:float=1e-6)->float:
    intersection = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    return float(intersection) / float(union+eps)

def select_slot_for_object(movi_mask_bool:np.ndarray, slot_masks:np.ndarray)->int:
    # select the slot that has the highest iou with the movi mask
    """
    movi_mask_bool: (H,W) bool
    slot_masks:     (num_slots,H,W) bool
    return:      best_slot_idx: int
    """
    best_iou = -1.0
    best_slot_idx = -1
    for slot_idx in range(slot_masks.shape[0]):
        slot_mask_bool = slot_masks[slot_idx]
        iou = iou_bool(movi_mask_bool, slot_mask_bool)
        if iou > best_iou:
            best_iou = iou
            best_slot_idx = slot_idx
    return best_slot_idx

def steve_visualize(video, recon_dvae, recon_tf, attns, num_slots,N=1):
    B, T, C, H, W = video.size()
    # B,T,K,C,H,W = attns.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :] # N,1,C,H,W
        # recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        # recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :] #N,K,1,H,W

        mask_video_t = video_t[:,0].unsqueeze(1) * attns_t.float()

        # tile
        tiles = torch.cat((video_t, mask_video_t), dim=1).flatten(end_dim=1)  # N*(K+1),C,H,W

        # grid
        frame = vutils.make_grid(tiles, nrow=(num_slots + 1), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames

class OCLModel:
    def __init__(self, ocl_path:str='/home/wz3008/slot-attn/steve', checkpoint_path:str="/checkpoint_path.pth"):
        sys.path.insert(0, ocl_path)
        from get_args import get_args
        from steve import STEVE
        
        self.ocl = STEVE(get_args())

        while ocl_path in sys.path:
            sys.path.remove(ocl_path)
        
        if os.path.isfile(checkpoint_path):
            print(f"Loading OCL checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.ocl.load_state_dict(checkpoint['model'])
        else:
            checkpoint = None

        if torch.cuda.is_available():
            self.ocl = self.ocl.cuda().eval()
        else:
            self.ocl = self.ocl.eval()

    @torch.no_grad()
    def infer(self, video: torch.Tensor):
        """
        intput:
          video: (B,T,3,H,W) torch.FloatTensor
        output:
          slots:      B, T, num_slots, slot_size torch.FloatTensor
          slot_masks: B, T, num_slots, 1, H, W torch.BoolTensor
        """
        video.cuda() if torch.cuda.is_available() else video
        slots, attns_vis, attns = self.ocl.encode(video)
        return slots, attns_vis, attns

def mask_bool(x: torch.Tensor) -> torch.Tensor:
    B, T, K, C, H, W = x.shape
    assert C == 1

    flat = x.view(B, T, K, H * W)

    for i in range(B):
        for j in range(T):
            for n in range(H*W):
                max_idx = flat[i, j, :, n].argmax()
                flat[i, j, :, n] = 0
                flat[i, j, max_idx, n] = 1
    return flat.view(B, T, K, 1, H, W)

def is_slot_mask_background(slot_mask:torch.Tensor):
    C, H, W = slot_mask.shape
    assert C == 1

    slot_sum = slot_mask.sum(dim=(1,2))  # sum over H,W
    area_frac = slot_sum / float(H * W)

    top = slot_mask[0,0,:].sum()
    bottom = slot_mask[0,-1,:].sum()
    left = slot_mask[0,:,0].sum()
    right = slot_mask[0,:,-1].sum()
    edge_sum = top + bottom + left + right

    if area_frac > 0.5 or edge_sum > 2*(H+W)*0.5:
        return True
    return False

def check_avg_movi_object_num(args):
    ds, ds_info = tfds.load(
        f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}",
        data_dir="gs://kubric-public/tfds",
        with_info=True,
    )
    data_iter = iter(tfds.as_numpy(ds[args.split]))

    total = 0
    count = 0
    max_obj = 0
    min_obj = 1e6
    for record in data_iter:
        # masks = record["segmentations"]  #(24, 128, 128, 1)
        # T = masks.shape[0]
        obj_num = len(record["instances"]['color_label'])
        count += obj_num
        total += 1
        max_obj = max(max_obj, obj_num)
        min_obj = min(min_obj, obj_num)
    print(f"Avg object num: {count/total}, max: {max_obj}, min: {min_obj}")
    # Avg object num: 6.500979078635473, max: 10, min: 3

def main(args):
    # get movi-* data and data info
    ds, ds_info = tfds.load(
        f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}",
        data_dir="gs://kubric-public/tfds",
        with_info=True,
    )
    data_iter = iter(tfds.as_numpy(ds[args.split]))

    to_tensor = transforms.ToTensor()

    #OCL model
    ocl_model = OCLModel(
        ocl_path = "/home/wz3008/slot-attn/steve",
        checkpoint_path = "/home/wz3008/slot-attn/steve_movi_a_2025-12-12T04:20:31.117982/checkpoint.pt.tar"
    )

    out_root = Path(args.out_path)
    os.makedirs(out_root, exist_ok=True)
    index_path = out_root / f"index_movi_{args.level}_{args.split}.jsonl"
    f_index = open(index_path, "w", encoding="utf-8")

    b = 0
    for record in tqdm(data_iter):
        # get_object_caption(record, ds_info, 0) -> 'large yellow rubber cube'
        video = record['video']   #(24, 128, 128, 3)
        masks = record["segmentations"]  #(24, 128, 128, 1)    index mask!
        T, *_ = video.shape
        
        video_torch = torch.from_numpy(video.astype(np.float32) / 255.0)  # (T,H,W,3)
        video_torch = video_torch.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
        video_torch = video_torch.unsqueeze(0) # (1,T,3,H,W)
        
        # import pdb; pdb.set_trace()
        if torch.cuda.is_available():
            video_torch = video_torch.cuda()

        slots, vis_attns, attns = ocl_model.infer(video_torch) # slots: (1,T,num_slots,slot_size), attns: (1,T,num_slots,1,H,W)
        
        slots = slots.detach().cpu()
        attns = attns.detach().cpu()
        
        slot_masks = mask_bool(attns) #B, T, K, 1, H, W
        for k in range(slot_masks.shape[2]):
            if is_slot_mask_background(slot_masks[0,0,k]):
                slot_masks[:,:,k,:,:,:] = 0
        slot_masks = slot_masks.squeeze(0)  # T, K, 1, H, W

        # if attns.ndim == 6:
        #     attn_maps = attns[0, :, :, 0, :, :] # (T,num_slots,H,W) B,T,K, 1, H, W
        # elif attns.ndim ==5:
        #     attn_maps = attns[0, :, :, :, :] # (T,num_slots,H,W)
        # else:
        #     raise ValueError("attns has wrong ndim")
        
        # slot_masks_bool = (attn_maps > 0.5).numpy().astype(bool)  # (T,S,H,W)
        # slot_masks = attns[0, :, :, 0, :, :].numpy().astype(np.float32)  # (T,S,H,W)
        slot_masks = slot_masks.numpy().astype(np.float32)  # (T,K,1,H,W)

        path_vid = os.path.join(args.out_path, f"{b:08}")
        os.makedirs(path_vid, exist_ok=True)

        path_slot = os.path.join(args.out_path, f"{b:08}_slot")
        os.makedirs(path_slot, exist_ok=True)

        for t in range(T):
            img = video[t]
            img = to_tensor(img)
            img_path = os.path.join(path_vid, f"{t:08}_image.png")
            vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))  #save image

            # mask = masks[t]  #(128, 128, 1)

            mask_num = len(np.unique(masks[t])) - 1  #background is 0

            for i in range(1,mask_num+1): #instance index starts from 1
                mask = (masks[t] == i).astype(float)
                mask_path = os.path.join(path_vid, f'{t:08}_mask_{i:02}.png')
                vutils.save_image(torch.Tensor(mask).permute(2, 0, 1), mask_path)

                slot_masks_t = slot_masks[t] # K, 1, H, W
                # best_slot_idx = select_slot_for_object(mask, slot_masks_t)
                best_slot_idx = select_slot_for_object(mask.squeeze(-1).astype(bool), slot_masks_t)
                # import pdb; pdb.set_trace()

                

                slot_feat = slots[0, t, best_slot_idx]
                slot_feat_path = os.path.join(path_slot, f"frame_{t:08}_inst_{i:03d}_slotfeat.pt")
                torch.save(slot_feat, slot_feat_path)

                slot_mask_path = os.path.join(path_slot, f"frame_{t:08}_inst_{i:03d}_slotmask.png")
                slot_mask_f = slot_masks_t[best_slot_idx].astype(np.float32)  # (H,W), [0,1]
                vutils.save_image(
                    torch.from_numpy(slot_mask_f).unsqueeze(0),  # (1,H,W)
                    slot_mask_path
                )

                # slot_mask_u8 = (slot_masks_t[best_slot_idx].astype(np.uint8) * 255)
                # slot_mask_path = os.path.join(path_slot, f"inst_{i:03d}_slotmask.png")
                # vutils.save_image(
                #     torch.Tensor(slot_mask_u8).unsqueeze(0),
                #     slot_mask_path
                # )

                caption = get_object_caption(record, ds_info, i-1)

                out_record = {
                    "video_idx": int(b),
                    "frame_idx": int(t),
                    "instance_idx": int(i),
                    "image_path": os.path.relpath(img_path, args.out_path),
                    "movi_mask_path": os.path.relpath(mask_path, args.out_path),
                    "selected_slot_idx": int(best_slot_idx),
                    "slot_feat_path": os.path.relpath(slot_feat_path, args.out_path),
                    "slot_mask_path": os.path.relpath(slot_mask_path, args.out_path),
                    "caption": caption,
                }
                f_index.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        b += 1

    f_index.close()
    print(f"Done. Wrote index to: {index_path}")

def load_mask(path):
    """
    Load a mask image and convert to boolean (H,W).
    """
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    return mask > 0

def load_image(path):
    """
    Load RGB image as float array in [0,1].
    """
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0

def overlay_mask(image, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
    """
    Overlay a boolean mask on an RGB image.

    image: (H,W,3), float in [0,1]
    mask:  (H,W), bool
    """
    out = image.copy()
    for c in range(3):
        out[..., c] = np.where(
            mask,
            (1 - alpha) * out[..., c] + alpha * color[c],
            out[..., c],
        )
    return out

def apply_mask(image, mask):
    """
    Apply a boolean mask as a stencil to an RGB image.
    Keep pixels where mask==True, set others to black.

    image: (H,W,3) float in [0,1]
    mask:  (H,W) bool
    return: (H,W,3) float in [0,1]
    """
    out = image.copy()
    out[~mask] = 0.0
    return out

def visualize(video_idx: int, frame_idx: int, out_root: str):
    """
    Visualize all object instances in a given (video_idx, frame_idx),
    and save the final figure to local disk.

    Args:
        video_idx: int
        frame_idx: int
        out_root:  root directory where images/jsonl are stored
    """
    out_root = Path(out_root)
    index_path = next(out_root.glob("index_movi_*_*.jsonl"))

    # ---------------------------------------------------------
    # Load all records matching (video_idx, frame_idx)
    # ---------------------------------------------------------
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["video_idx"] == video_idx and rec["frame_idx"] == frame_idx:
                records.append(rec)

    if len(records) == 0:
        print(f"No records found for video {video_idx}, frame {frame_idx}")
        return

    # ---------------------------------------------------------
    # Load original image (shared by all instances)
    # ---------------------------------------------------------
    img_path = out_root / records[0]["image_path"]
    image = load_image(img_path)

    # ---------------------------------------------------------
    # slot mask visualization
    # ---------------------------------------------------------
    # import pdb; pdb.set_trace()
    ocl_model = OCLModel(
        ocl_path = "/home/wz3008/slot-attn/steve",
        checkpoint_path = "/home/wz3008/slot-attn/steve_movi_a_2025-12-12T04:20:31.117982/checkpoint.pt.tar"
    )
    img = Image.open(img_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    video = to_tensor(img).unsqueeze(0).unsqueeze(0).cuda()
    slots, attns_vis, attns = ocl_model.infer(video)
    slot_mask = mask_bool(attns) #B, T, K, 1, H, W
    for k in range(slot_mask.shape[2]):
        if is_slot_mask_background(slot_mask[0,0,k]):
            slot_mask[:,:,k,:,:,:] = 0
    save_path = os.path.join("/home/wz3008/slot-attn", f"steve_vis.png")
    frames = steve_visualize(video, None, None, slot_mask, slot_mask.shape[2],N=1)
    vutils.save_image(frames.squeeze(0), save_path)



    num_objs = len(records)

    fig, axes = plt.subplots(
        num_objs, 3, figsize=(12, 4 * num_objs), squeeze=False
    )

    for i, rec in enumerate(records):
        movi_mask_path = out_root / rec["movi_mask_path"]
        slot_mask_path = out_root / rec["slot_mask_path"]

        movi_mask = load_mask(movi_mask_path)
        slot_mask = load_mask(slot_mask_path)

        img_movi = apply_mask(image, movi_mask)
        img_slot = apply_mask(image, slot_mask)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original\n(inst {rec['instance_idx']})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_movi)
        axes[i, 1].set_title("MOVi mask overlay")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_slot)
        axes[i, 2].set_title(f"Slot mask overlay\nCaption: {rec['caption']}")
        axes[i, 2].axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save the figure to local disk (under out_root)
    # ---------------------------------------------------------
    save_path = out_root / f"viz_video{video_idx:08d}_frame{frame_idx:08d}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--level", type=str, default="a", help="MOVi level: a/b/c/d/e")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument('--split', default='train')
    parser.add_argument('--version', default='1.0.0')
    parser.add_argument("--out_path", type=str, default="/home/wz3008/slot-attn/output/")

    args = parser.parse_args()
    # main(args)
    visualize(0, 0, "/home/wz3008/slot-attn/output")
    # check_avg_movi_object_num(args)
