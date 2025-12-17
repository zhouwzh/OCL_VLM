import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from datetime import datetime

from steve import STEVE
from data import GlobVideoDataset, SAYCAMDataset
from utils import cosine_anneal, linear_warmup

print("PyTorch Version: ", torch.__version__)