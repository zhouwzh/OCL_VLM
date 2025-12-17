import math
import os.path
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from steve import STEVE
from data import GlobVideoDataset, SAMCAMDataset, SAYCAMDataset
from utils import * #cosine_anneal, linear_warmup

import glob
from PIL import Image
from torchvision import transforms
import json
from tqdm import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=6)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='/home/wz3008/dataset/movi-e/*')
parser.add_argument('--log_path', default='/home/wz3008/steve/logs/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=200000)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=False, action='store_true')

parser.add_argument('--dev', action='store_true')
parser.add_argument('--use_backbone', action='store_true')
parser.add_argument('--use_dvae', action='store_true')
parser.add_argument('--backbone_arch', type=str, default='resnext50_32x4d')
parser.add_argument('--backbone_checkpoint', type=str, default='/home/wz3008/logs/movie_dino_resnext50_0080.pth')
parser.add_argument('--dataset', type=str, default='movie')
parser.add_argument('--visual_output', type=str, default='/home/yy2694/scratch/SlotAttn/steve/visualize_frames')
parser.add_argument('--visual_num',type=int,default=50)

parser.add_argument('--json_path', default='/scratch/yy2694/data/saycam/saycam_transcript_frames/')
parser.add_argument('--start_epoch', type=int, default=0)


args = parser.parse_args()

VISUAL_OUTPUT_TRAIN = os.path.join(args.visual_output,"tarin")
VISUAL_OUTPUT_TEST = os.path.join(args.visual_output,"test")
VISUAL_OUTPUT_EVAL = os.path.join(args.visual_output,"eval")
os.makedirs(VISUAL_OUTPUT_TRAIN, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_TEST, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_EVAL, exist_ok=True)

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)

SAYCAM_DATA_DIR = "/mnt/wwn-0x5000c500e421004a/yy2694/datasets/train_5fps"
if args.dataset == 'movie':
    train_dataset = GlobVideoDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
    val_dataset = GlobVideoDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
    eval_dataset = GlobVideoDatasetWithMasks(root='/home/wz3008/dataset/movi-e-eval/*', img_size=args.image_size, num_segs=25,ep_len=args.ep_len, img_glob='????????_image.png')
elif args.dataset == 'saycam':
    train_dataset = SAYCAMDataset(img_dir=args.data_path,json_path=args.json_path, phase="train",img_size=128)
    val_dataset = SAYCAMDataset(img_dir=args.data_path,json_path=args.json_path, phase="val",img_size=128)
    eval_dataset = datasets.ImageFolder(
        root="/scratch/yy2694/data/saycam/saycam_labeled",
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
    )
elif args.dataset == 'samcam':
    train_dataset = SAMCAMDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='*.jpg')
    val_dataset = SAMCAMDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='*.jpg')



loader_kwargs = {'batch_size': args.batch_size,'shuffle': True,'num_workers': args.num_workers,'pin_memory': True,'drop_last': True,}
train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)
train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

if args.dataset == 'movie':
    eval_loader = DataLoader(eval_dataset,sampler=None, **loader_kwargs)
    eval_epoch_size = len(eval_loader)
elif args.dataset == 'saycam':
    eval_loader = DataLoader(eval_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    eval_epoch_size = len(eval_loader)

# log_interval = train_epoch_size // 5

model = STEVE(args)
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
start_epoch = args.start_epoch #checkpoint['epoch']
# best_val_loss = checkpoint['best_val_loss']
# best_epoch = checkpoint['best_epoch']
model.load_state_dict(checkpoint)
model = model.cuda().eval()

def visualize(video, recon_dvae, recon_tf, attns, N=1):
    # video: 1, 1, 3, 128, 128
    # recon_dvae: 1,1,3,128,128
    # attns: 1,1,6,3,128,128
    # import pdb; pdb.set_trace()
    B, T, C, H, W = video.size()

    frames = []
    attns = add_border_to_attn_images(attns, color=(0., 0., 0.), thickness_px=1, thr=0.05, use_contour=True, use_bbox=False)  # (N, T, num_slots, 3, 128, 128)
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]  # 4, 1, 3, 128, 128
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :] # 4, 15, 3, 128, 128

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 3), pad_value=0.8)  # 3, 132, 2342
        frames += [frame]

    # import pdb; pdb.set_trace()
    # frames = torch.stack(frames, dim=0).unsqueeze(0)
    frames = torch.cat(frames, dim=1).unsqueeze(0)

    return frames

with torch.no_grad():
    global_step = start_epoch * train_epoch_size
    tau = cosine_anneal(global_step,args.tau_start,args.tau_final,0,args.tau_steps)
    # import pdb; pdb.set_trace()

    # for batch, video in tqdm(enumerate(train_loader)): 
    #     video = video.cuda()
    #     (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)
    #     gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
    #     frames = visualize(video, recon, gen_video, attns, N=8)
    #     vutils.save_image(frames, f"{VISUAL_OUTPUT_TRAIN}/{batch}.png")
    #     print(f"saved {VISUAL_OUTPUT_TRAIN}/{batch}.png")
            
    #     if args.dev or batch >args.visual_num: break
    
    # for batch, video in tqdm(enumerate(val_loader)):
    #     video = video.cuda()
    #     (recon, cross_entropy, mse, attns) = (model.module if args.use_dp else model).visualize_recon(video, tau, args.hard)
    #     gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
    #     frames = visualize(video, recon, gen_video, attns, N=8)
    #     vutils.save_image(frames, f"{VISUAL_OUTPUT_TEST}/{batch}.png")
    #     print(f"saved {VISUAL_OUTPUT_TEST}/{batch}.png")

    #     if args.dev or batch >args.visual_num: break

    # for batch,(video,_) in tqdm(enumerate(eval_loader)):
    #     video = video.cuda()
    #     if args.dataset == 'saycam':
    #         video = video.unsqueeze(1)
    #     (recon, cross_entropy, mse, attns) = (model.module if args.use_dp else model).visualize_recon(video.unsqueeze(1), tau, args.hard)
    #     gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
    #     frames = visualize(video, recon, gen_video, attns, N=8)
    #     vutils.save_image(frames, f"{VISUAL_OUTPUT_EVAL}/{batch}.png")
    #     print(f"saved {VISUAL_OUTPUT_EVAL}/{batch}.png")

    #     if args.dev or batch >args.visual_num: break

    # for folder in os.listdir("/scratch/yy2694/data/saycam/saycam_labeled"):
    #     if folder not in ['ball','cat','car']: continue
    #     files = sorted(os.listdir(os.path.join("/scratch/yy2694/data/saycam/saycam_labeled", folder)))
    #     id = 0
    #     for i in range(0,len(files)):
    #         image_path = files[i]
    #         id += 1
    #         image = Image.open(os.path.join("/scratch/yy2694/data/saycam/saycam_labeled",folder,image_path)).convert("RGB")
    #         transform = transforms.Compose([
    #             transforms.Resize((128, 128)),
    #             transforms.ToTensor()
    #         ])
    #         image = transform(image).unsqueeze(0).cuda() #(1,3,128,128)
    #         (recon, cross_entropy, mse, attns) = (model.module if args.use_dp else model).visualize_recon(image.unsqueeze(1), tau, args.hard)
    #         gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(image.unsqueeze(1)[:8])
    #         frames = visualize(image.unsqueeze(1), recon, gen_video, attns, N=1)
    #         os.makedirs(os.path.join(VISUAL_OUTPUT_EVAL,folder), exist_ok=True)
    #         vutils.save_image(frames, f"{VISUAL_OUTPUT_EVAL}/{folder}/{image_path[:-5]}.png")
    #         print(f"saved {VISUAL_OUTPUT_EVAL}/{folder}/{image_path[:-5]}.png")
    #         # import pdb; pdb.set_trace()

    #         if id >args.visual_num: break
    
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
    frames = []
    for path in file_paths:
        image = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0).cuda() #(1,3,128,128)
        (recon, cross_entropy, mse, attns) = (model.module if args.use_dp else model).visualize_recon(image.unsqueeze(1), tau, args.hard)
        # (recon, cross_entropy, mse, attns) =  model(image.unsqueeze(1), tau, args.hard)
        gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(image.unsqueeze(1)[:8])
        frame = visualize(image.unsqueeze(1), recon, gen_video, attns, N=1)
        frames.append(frame)
        print(f"processed {path}")
    frames = torch.cat(frames, dim=2)
    vutils.save_image(frames, f"/home/yy2694/scratch/SlotAttn/steve/visualize_frames/{Path(args.checkpoint_path).stem}.png")
    print(f"saved /home/yy2694/scratch/SlotAttn/steve/visualize_frames/{Path(args.checkpoint_path).stem}.png")


    # folder = path.split("/")[-2]
    # image_path = path.split("/")[-1]
    # os.makedirs(os.path.join(VISUAL_OUTPUT_EVAL,folder), exist_ok=True)
    # vutils.save_image(frames, f"{VISUAL_OUTPUT_EVAL}/{folder}/{image_path[:-4]}_v2.png")
    # print(f"saved {VISUAL_OUTPUT_EVAL}/{folder}/{image_path[:-4]}_v2.png")

