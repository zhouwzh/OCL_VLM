import os
import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from pathlib import Path

ROOT = "/scratch/wz3008/data/saycam_labeled"
file_paths = [
    os.path.join(ROOT,'cat','img_101129.jpeg'),
    os.path.join(ROOT,'cat','img_101158.jpeg'),
    os.path.join(ROOT,'cat','img_103032.jpeg'),
    os.path.join(ROOT,'car','img_10136.jpeg'),
    os.path.join(ROOT,'car','img_141101.jpeg'),
    os.path.join(ROOT,'car','img_141113.jpeg'),
    os.path.join(ROOT,'ball','img_102606.jpeg'),
    os.path.join(ROOT,'ball','img_107105.jpeg'),
    os.path.join(ROOT,'ball','img_108679.jpeg'),
]

def visualize_fn(image, attent):
    _,C,H,W = image.size()
    _,K,_,_ = attent.size()
    attent = F.interpolate(attent, size=(H, W), mode='bilinear', align_corners=False) # 1,7,256,256
    # attns = add_border_to_attn_images(attns, color=(0., 0., 0.), thickness_px=1, thr=0.05, use_contour=True, use_bbox=False)  # (N, T, num_slots, 3, 128, 128)
    best_slot_id = attent.argmax(dim=1).unsqueeze(1) # 1,1,256,256
    slot_mask = (best_slot_id.unsqueeze(1) == torch.arange(K).to(image.device).view(1,K,1,1,1)) # 1,K,1,H,W
    slot_mask = slot_mask.float()
    attn_t = image.unsqueeze(0).unsqueeze(0) * slot_mask + (1. - slot_mask) #1,1, 7, 3, 256, 256
    tiles = torch.cat((image.unsqueeze(0),attn_t[0]),dim=1).flatten(end_dim=1) # 1+K, C,H,W
    frame = vutils.make_grid(tiles,nrow=(1+K),pad_value=0.8)
    return frame
    
# def TTT(model, image, step):
    
    

def visualize(args,model):
    model.eval().cuda()
    frames1 = []
    frames2 = []
    for path in file_paths:
        image = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0).cuda() # 3, 256, 256

        feature, slotz, attent, attent2, recon = model(image) #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
        # import pdb; pdb.set_trace()
        frame1 = visualize_fn(image, attent)
        frame2 = visualize_fn(image, attent2)
        frames1 += [frame1]
        frames2 += [frame2]
    frames1 = torch.cat(frames1,dim=1)
    frames2 = torch.cat(frames2,dim=1)
    vutils.save_image(frames1, f"/home/yy2694/scratch/SlotAttn/MetaSlot/visualize_frames/{os.path.basename(args.cfg_file)[:-3]}_1.png")
    print(f"saved /home/yy2694/scratch/SlotAttn/MetaSlot/visualize_frames/{os.path.basename(args.cfg_file)[:-3]}_1.png")
    vutils.save_image(frames2, f"/home/yy2694/scratch/SlotAttn/MetaSlot/visualize_frames/{os.path.basename(args.cfg_file)[:-3]}_2.png")
    print(f"saved /home/yy2694/scratch/SlotAttn/MetaSlot/visualize_frames/{os.path.basename(args.cfg_file)[:-3]}_2.png")
    