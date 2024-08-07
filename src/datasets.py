import os
import torch
from torchvision import datasets

class AG_NEWS(Dataset):
  """AG_NEWS dataset."""

  def __init__(self, data_dir="", split="train"):
    """
    Arguments:
        data_dir (string): Path to the data folder. Default="".
        split (string): Desired data split. Default="train.
    """
    self.ds_dir = os.path.join(data_dir, f"AG_NEWS_{split}.pkl")
    self.split = split
    self.ds = None
    self.__read_ds()
    self.text = self.ds[0]
    self.target = self.ds[1]
    self.data = self.ds[2]

  def __read_ds(self):
    with open(self.ds_dir, 'rb') as f:
      self.ds = pkl.load(f)

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
    if torch.is_tensor(idx): idx = idx.tolist()
    return self.data[idx], self.target[idx]

  def get_full_item(self, idx):
    if torch.is_tensor(idx): idx = idx.tolist()
    return self.text[idx], self.data[idx], self.target[idx]