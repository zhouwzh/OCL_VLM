import argparse
import torch, torch.nn as nn
from steve import STEVE

def set_args_for_steve():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--ep_len', type=int, default=3)

    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
    parser.add_argument('--data_path', default='/scratch/wz3008/cvcl-related/movi-e/*')
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

    parser.add_argument('--vocab_size', type=int, default=2048)
    parser.add_argument('--num_decoder_blocks', type=int, default=8)
    parser.add_argument('--num_decoder_heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_steps', type=int, default=30000)

    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--use_dp', default=True, action='store_true')

    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--use_backbone', action='store_true')
    parser.add_argument('--use_dvae', action='store_true')
    parser.add_argument('--backbone_arch', type=str, default='resnext50_32x4d')
    parser.add_argument('--backbone_checkpoint', type=str, default='/home/wz3008/dino/output/checkpoint0080.pth')

    parser.add_argument('--dataset', type=str, default='movie')

    args = parser.parse_args(args=[])
    return args

def load_steve(ckpt_path, backbone, dvae):
    steve_args = set_args_for_steve()
    steve_args.use_backbone = backbone
    steve_args.use_dvae = dvae
    steve = STEVE(steve_args)
    if ckpt_path is not None:
        print(f"Loading STEVE from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        steve.load_state_dict(checkpoint['model'])
    return steve