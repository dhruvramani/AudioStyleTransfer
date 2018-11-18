import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import torchaudio
from vctk import VCTK
from feature import *

vocab = None

def set_vocab(new_vocab):
    global vocab
    vocab = new_vocab

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data = list(filter(lambda x: type(x[1]) != int, data))
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def target_transform(targets):
    global vocab
    tokens = nltk.tokenize.word_tokenize(str(targets[0]).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)

    return target

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    stft = transform_stft(inp)
    stft = torch.Tensor(stft)
    stft = stft.unsqueeze(0)
    return stft


def get_loader(root="~/dataset/", batch_size=50, shuffle=True, num_workers=2, transforms=True):
    """Returns torch.utils.data.DataLoader for custom VCTK dataset."""
    if(transforms):
        vctk_dataset = VCTK(root, download=False, transform=inp_transform, target_transform=target_transform)
        train_size = int(0.8 * len(vctk_dataset))
        test_size = len(vctk_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(vctk_dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        return train_loader, test_loader
    else :
        vctk_dataset = VCTK(root, download=False)
        data_loader = torch.utils.data.DataLoader(dataset=vctk_dataset, batch_size=1, shuffle=shuffle, num_workers=1)
        return data_loader, None
