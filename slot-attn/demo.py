import sys, os
import json
import torch
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as transforms

ocl_path:str='/home/wz3008/slot-attn/steve'
checkpoint_path:str="/home/wz3008/slot-attn/steve_movi_a_2025-12-12T04:20:31.117982/checkpoint.pt.tar"
sys.path.insert(0, ocl_path)
from get_args import get_args
from steve import STEVE

def visualize(video, recon_dvae, recon_tf, attns, num_slots,N=1):
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

    # max_indices = flat.argmax(dim=2) # (B, T, N)
    # out_flat = torch.zeros_like(flat)  # (B, T, K, N)
    # out_flat.scatter_(dim=3, index=max_indices.unsqueeze(2), value=1.0)


steve = STEVE(get_args())
checkpoint = torch.load(checkpoint_path, map_location='cpu')
steve.load_state_dict(checkpoint['model'])

img_path = "/home/wz3008/slot-attn/output/00000000/00000000_image.png"
img = Image.open(img_path).convert("RGB")
to_tensor = transforms.ToTensor()
video = to_tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

video = video.cuda()
steve = steve.cuda()




# (recon, cross_entropy, mse, attns) = steve(video, 1, None)

# gen_video = steve.reconstruct_autoregressive(video[:8])
# print(attns.shape)

# frames = visualize(video, recon, gen_video, attns, attns.shape[2],N=8)
# vutils.save_image(frames.flatten(end_dim=1), 'reconstruction.png', nrow=frames.size(2), pad_value=0.8)

# vutils.save_image(frames.squeeze(0), save_path)


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

slots, attns_vis, attns = steve.encode(video)
slot_mask = mask_bool(attns) #B, T, K, 1, H, W
for k in range(slot_mask.shape[2]):
    if is_slot_mask_background(slot_mask[0,0,k]):
        slot_mask[:,:,k,:,:,:] = 0
        
save_path = os.path.join("/home/wz3008/slot-attn", f"steve_vis.png")
frames = visualize(video, None, None, slot_mask, slot_mask.shape[2],N=1)
vutils.save_image(frames.squeeze(0), save_path)












