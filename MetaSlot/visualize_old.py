import os
import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from pathlib import Path

ROOT = "/scratch/yy2694/data/saycam/saycam_labeled"
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

import copy
def TTT(model, image, step):
    backup_state_dict = copy.deepcopy(model.state_dict())
    params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encode_backbone.")
    ]
    optimizer = torch.optim.Adam(params, lr=2e-4)
    for _ in range(step):
        model.train()
        feature, slotz, attent, attent2, recon = model(image)
        loss = F.mse_loss(recon, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(backup_state_dict)
    model.eval()

    with torch.no_grad():
        feature, slotz, attent, attent2, recon = model(image)
    return feature, slotz, attent, attent2, recon



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

        if not args.visualize_ttt:
            feature, slotz, attent, attent2, recon = model(image) #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
        else:
            feature, slotz, attent, attent2, recon = TTT(model, image, step = 4)
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


# def _edge_from_mask(mask_bool, k=3):
#     N, T, S, _, H, W = mask_bool.shape
#     m = mask_bool.float().view(N*T*S, 1, H, W)
#     dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2)  #360, 1, 128, 128
#     ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=k//2)
#     edge = (dil - ero) > 0
#     return edge.view(N, T, S, 1, H, W)

# def add_border_to_attn_images(attns, 
#                               color,  # RGB [0,1]
#                               thickness_px, 
#                                 thr,
#                               use_contour=True,
#                               use_bbox=False):
#     """
#     video: (N,1,3,128,128)
#     attns: (N, T, num_slots, 3, 128, 128)
#     """
#     # import pdb; pdb.set_trace()
#     N,T,S,C,H,W = attns.shape
#     device = attns.device
#     dtype = attns.dtype

#     # attns_img = video.squeeze(1) * attns_mask_up + (1. - attns_mask_up)  # (N, num_slots, 3, 128, 128)

#     mask_gray = 1.0 - attns.mean(dim=3, keepdim=True)  # (N, T, S, 1, H, W)
#     mask_bin = (mask_gray > 0) # (N, T, S, 1, H, W)

#     attns_img = attns.clone()

#     edge = _edge_from_mask(mask_bin, k=3)
    
#     ks = 2*thickness_px - 1 if thickness_px > 1 else 1
#     if ks > 1:
#         e = edge.float().view(N*T*S, 1, H, W)
#         e = torch.nn.functional.max_pool2d(e, kernel_size=ks, stride=1, padding=thickness_px//2)
#         edge = (e > 0).view(N, T, S, 1, H, W)
    
#     col = torch.tensor(color, dtype=dtype, device=device).view(1,1,1,3,1,1)  # (1,1,1,C,1,1)
#     edge_c = edge.expand(N,T,S,C,H,W)   # (B,T,S,C,H,W)
#     attns_img = attns_img.clone()
#     attns_img[edge_c] = col.expand_as(attns_img)[edge_c]

#     return attns_img
                

