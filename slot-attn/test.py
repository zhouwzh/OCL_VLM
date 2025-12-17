import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import movi_a

ds, ds_info = tfds.load(
    "movi_a",  
    data_dir="/home/wz3008/slot-attn/tfds_data",
    # data_dir="gs://kubric-public/tfds",
    with_info=True,
    download=True,
)
