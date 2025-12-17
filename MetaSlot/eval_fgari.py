import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from pathlib import Path
import copy

import torchvision.utils as vutils
import torch.nn.functional as F
from scipy.special import comb


class LabeledSDatasetWithMasks(Dataset):
    def __init__(self, root, img_size, num_segs, ep_len=1, img_glob='*_image.png'): #img_8487_mask_08.png
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
            # image_paths = image_paths[:(len(image_paths)//23)]
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

def compute_ari(table):
    """
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """

    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()

    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()

    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
                (comb_table - comb_a * comb_b / comb_n) /
                (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )

    return ari


def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, D)
    :param mask1: predicted mask, (N1, D)
    :return:
    """

    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, D)
    mask0 = mask0[:, None].byte()
    # (1, N1, D)
    mask1 = mask1[None, :].byte()
    # (N0, N1, D)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1)

    return compute_ari(table.numpy())


def evaluate_ari(true_mask, pred_mask):
    """
    :param
        true_mask: (B, N0, D)
        pred_mask: (B, N1, D)
    :return: average ari
    """
    from torch import arange as ar

    B, K, D = pred_mask.size()

    if pred_mask.dtype == torch.bool:
        pred_mask = pred_mask.float()
    # max_index (B, D)
    max_index = torch.argmax(pred_mask, dim=1)

    # get binarized masks (B, N1, D)
    pred_mask = torch.zeros_like(pred_mask)
    pred_mask[ar(B)[:, None], max_index, ar(D)[None, :]] = 1.0

    aris = 0.
    for b in range(B):
        aris += compute_mask_ari(true_mask[b].detach().cpu(), pred_mask[b].detach().cpu())

    avg_ari = aris / B
    return avg_ari


def TTT(model, image, step, ttt_lr):
    backup_state_dict = copy.deepcopy(model.state_dict())

    is_dp = isinstance(model, torch.nn.DataParallel)
    backbone_prefix = "module.encode_backbone." if is_dp else "encode_backbone."

    params = [
        p for name, p in model.named_parameters()
        if not name.startswith(backbone_prefix)
    ]
    optimizer = torch.optim.Adam(params, lr=ttt_lr)

    ret = []

    # print("=====TTT Start=====")
    for i in range(step):
        model.train()
        feature, slotz, attent, attent2, recon = model(image)
        loss = F.mse_loss(feature, recon)

        optimizer.zero_grad()
        loss.backward()
        # print(f"loss: {loss.item()}")
        optimizer.step()

        # if i % 2 == 0:
        ret.append({
            "step": i,
            "attent2": attent2.detach().cuda(),
        })
    # print("=====TTT Finish=====")
    model.eval()

    with torch.no_grad():
        feature, slotz, attent, attent2, recon = model(image)
    model.load_state_dict(backup_state_dict)
    # return feature, slotz, attent, attent2, recon
    return ret

def write_log(log_file, msg):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

def eval(args,model):
    log_file = 'full_dataset_probe_eval_log.txt'
    model = model.eval()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        print("Using CPU!")
    
    print(f"Evaluating without TTT")
    write_log(log_file, f"Evaluating without TTT")
    eval_lr(args,model,lr=0.0,log_file=log_file,ttt=False)

          
    #for lr in [1e-52e-05, 6e-5, 8e-5, 1e-4, 1.2e-4, , 8e-6, 6e-6, 4e-6, 2e-6]:#2e-5, 6e-5, 8e-5, 1e-4, 1.2e-4]:

    # for lr in [2e-5, 8e-5, 6e-6, 8e-6, 6e-5, 2e-6, 1e-4, 4e-6, 1.2e-4]: # 1e-05
    #     print(f"Evaluating TTT with lr={lr}")
    #     write_log(log_file, f"Evaluating TTT with lr={lr}")
    #     eval_lr(args,model,lr,log_file,ttt=True)

def eval_lr(args,model,lr,log_file,ttt=True):
    eval_dataset = LabeledSDatasetWithMasks(
        root='/masks_output/*',
        img_size=256,
        num_segs=25,
        ep_len=1,
        img_glob='*_image.png'
    )
    # len(eval_dataset) = 5787
    
    if ttt:
        batch_size = 1
    else:
        batch_size = 64

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    eval_loader = DataLoader(eval_dataset, sampler=None, **loader_kwargs)

    # model.eval().cuda()
    
    # import pdb; pdb.set_trace()
    # with torch.no_grad():
    fgaris = {}
    fgari_no_ttt = []
    for batch, (image, true_masks) in enumerate(eval_loader):
        image = image.cuda()  # B, 3, H, W
        B, T, C, H, W = image.size()

        if not ttt:
            feature, slotz, attent, attent2, recon = model(image.squeeze(1)) #1,768,14,14 | 1,7,256 | 1,7,14,14 | 1,7,14,14 | 1,768,14,14
            _,K,_,_ = attent2.size()
            #B,7,14,14 -> B,num_slots,1,H,W
            attent = F.interpolate(attent2, size=(H, W), mode='bilinear', align_corners=False) # 1,7,256,256
            # attns = add_border_to_attn_images(attns, color=(0., 0., 0.), thickness_px=1, thr=0.05, use_contour=True, use_bbox=False)  # (N, T, num_slots, 3, 128, 128)
            best_slot_id = attent.argmax(dim=1).unsqueeze(1) # 1,1,256,256
            slot_mask = (best_slot_id.unsqueeze(1) == torch.arange(K).to(image.device).view(1,K,1,1,1)) # B,K,1,H,W
            # to B,T,K,1,H,W
            pred_masks_b_m = slot_mask.unsqueeze(1)
            fgari_b_m = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                                            pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
            fgari_no_ttt += [fgari_b_m]
        else:
            outputs = TTT(model, image.squeeze(1), args.ttt_step, lr)
            _,K,_,_ = outputs[0]['attent2'].size()

            # fgaris_b = []
            for out in outputs:
                step = out['step']
                attent2 = out['attent2']

                #B,7,14,14 -> B,num_slots,1,H,W
                attent = F.interpolate(attent2, size=(H, W), mode='bilinear', align_corners=False) # 1,7,256,256
                # attns = add_border_to_attn_images(attns, color=(0., 0., 0.), thickness_px=1, thr=0.05, use_contour=True, use_bbox=False)  # (N, T, num_slots, 3, 128, 128)
                best_slot_id = attent.argmax(dim=1).unsqueeze(1) # 1,1,256,256
                slot_mask = (best_slot_id.unsqueeze(1) == torch.arange(K).to(image.device).view(1,K,1,1,1)) # B,K,1,H,W
                # to B,T,K,1,H,W
                pred_masks_b_m = slot_mask.unsqueeze(1)
            

                fgari_b_m = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                                            pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
                # fgaris_b += [fgari_b_m]
                if step not in fgaris:
                    fgaris[step] = []
                fgaris[step] += [fgari_b_m]

        # if batch % 100 == 0:
        #     write_log(log_file, f"Processed {batch} batches.")
        # fgaris += [fgaris_b]
    # import pdb; pdb.set_trace()

    if not ttt:
        avg_fgari = sum(fgari_no_ttt) / len(fgari_no_ttt)
        print(f"FG-ARI without TTT: {avg_fgari:.2f}")
        write_log(log_file, f"FG-ARI without TTT: {avg_fgari:.2f}")
    else:
        for step in sorted(fgaris.keys()):
            avg_fgari = sum(fgaris[step]) / len(fgaris[step])
            print(f"Step {step} FG-ARI: {avg_fgari:.2f}")
            write_log(log_file, f"Step {step} FG-ARI: {avg_fgari:.2f}")

        # avg_fgari = sum(fgaris) / len(fgaris)
        # print(f"FG-ARI: {avg_fgari:.2f}")
