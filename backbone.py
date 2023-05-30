from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from argparse import Namespace

device = torch.device('cuda')

class RAFTTransformerBackbone(nn.Module):
    def __init__(self, args=None, max_iters=30, raft_hidden_size=128, image_size=[368, 768]):
        super().__init__()
        if args is None:
            args = Namespace(name='raft-sintel', stage='sintel', restore_ckpt=None, small=False,
                             validation=['sintel'], lr=0.0001, num_steps=120000, batch_size=5,
                             image_size=image_size, gpus=[0], mixed_precision=True, iters=12,
                             wdecay=1e-05, epsilon=1e-08, clip=1.0, dropout=0.0, gamma=0.85,
                             add_noise=False)

        self.raft = RAFT(args).to(device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=raft_hidden_size, nhead=4,
                                                            batch_first=True, device=device).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=3)
        self.feature_token = nn.Parameter(torch.rand(raft_hidden_size)).to(device)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True):
        # print(f"image1: {image1.shape}\t\timage2: {image2.shape}")
        _, hidden_states = self.raft(image1, image2, iters=iters, flow_init=flow_init,
                                     upsample=upsample, feature_mode=True)
        batch_size, _, h, w = hidden_states[0].shape
        for i, hidden_state in enumerate(hidden_states):
            hidden_states[i] = torch.permute(hidden_state, (0, 2, 3, 1))
        # print("#########", batch_size, h, w, _)
        class_tokens = self.feature_token.repeat(batch_size, h, w, 1)
        # print(f"hidden_states: {len(hidden_states), hidden_states[0].shape}\t\tres: {class_tokens.shape}")
        hidden_states.append(class_tokens)
        seq_len = len(hidden_states)
        hidden_states = torch.stack(hidden_states, dim=-2)
        # print(f"hidden_states: {hidden_states.shape}\t\tres: {class_tokens.shape}")

        c = hidden_states[0].shape[-1]
        hidden_states = hidden_states.view(-1, seq_len, c)
        # print("JESUS CHRIST", hidden_states.shape)
        res = self.transformer_encoder(hidden_states)[..., -1, :]
        return res.view(batch_size, h, w, -1).permute(0, 3, 1, 2)
    

    def forward_numpy(self, img1, img2, iters=12, flow_init=None, upsample=True):
        assert img1.shape == img2.shape, "Image shapes must match"
        assert len(img1.shape) == 3 or len(img1.shape) == 4, "Inputs should either be a single image, or a batch of them"
        CHANNEL_NP_TO_TORCH = (0, 3, 1, 2)

        image1 = torch.from_numpy(img1).to(device, dtype=torch.float32)
        image2 = torch.from_numpy(img2).to(device, dtype=torch.float32)
        if len(image1.shape) == 3:
            image1, image2 = torch.unsqueeze(image1, 0), torch.unsqueeze(image2, 0)
        image1 = torch.permute(image1, CHANNEL_NP_TO_TORCH)
        image2 = torch.permute(image2, CHANNEL_NP_TO_TORCH)
        
        features = self.forward(image1, image2, iters=iters, flow_init=flow_init, upsample=upsample)

        return features.detach().cpu().numpy()


class RAFTAvgBackbone(nn.Module):
    def __init__(self, args=None, max_iters=30, image_size=[368, 768]):
        super().__init__()
        if args is None:
            args = Namespace(name='raft-sintel', stage='sintel', restore_ckpt=None, small=False,
                             validation=['sintel'], lr=0.0001, num_steps=120000, batch_size=5,
                             image_size=image_size, gpus=[0], mixed_precision=True, iters=12,
                             wdecay=1e-05, epsilon=1e-08, clip=1.0, dropout=0.0, gamma=0.85,
                             add_noise=False)
        
        self.raft = RAFT(args).to(device)
        self.averaging_weights = torch.nn.Parameter(torch.rand(max_iters)).to(device)
    

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True):
        # print(f"image1: {image1.shape}\t\timage2: {image2.shape}")
        _, hidden_states = self.raft(image1, image2, iters=iters, flow_init=flow_init,
                                     upsample=upsample, feature_mode=True)
        weights = nn.functional.softmax(self.averaging_weights[:len(hidden_states)]) \
                .view(-1, *[1 for i in range(len(hidden_states[0].shape))])
        hidden_states = torch.stack(hidden_states)
        # print(f"hidden_states: {hidden_states.shape}\t\tres: {weights.shape}")
        hidden_states *= weights
        res = torch.mean(hidden_states, dim=0)
        # print(f"hidden_states: {hidden_states.shape}\t\tres: {res.shape}")
        return res
    

    def forward_numpy(self, img1, img2, iters=12, flow_init=None, upsample=True):
        assert img1.shape == img2.shape, "Image shapes must match"
        assert len(img1.shape) == 3 or len(img1.shape) == 4, "Inputs should either be a single image, or a batch of them"
        CHANNEL_NP_TO_TORCH = (0, 3, 1, 2)

        image1 = torch.from_numpy(img1).to(device, dtype=torch.float32)
        image2 = torch.from_numpy(img2).to(device, dtype=torch.float32)
        if len(image1.shape) == 3:
            image1, image2 = torch.unsqueeze(image1, 0), torch.unsqueeze(image2, 0)
        image1 = torch.permute(image1, CHANNEL_NP_TO_TORCH)
        image2 = torch.permute(image2, CHANNEL_NP_TO_TORCH)
        
        features = self.forward(image1, image2, iters=iters, flow_init=flow_init, upsample=upsample)

        return features.detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # Sanity Checks

    img1 = np.random.random((368, 768, 3))
    img2 = np.random.random((368, 768, 3))

    backbone = RAFTAvgBackbone().to(device)
    backbone.eval()
    features = backbone.forward_numpy(img1, img2)
    print(features.shape)
    
    backbone = RAFTTransformerBackbone()
    features = backbone.forward_numpy(img1, img2)
    print(features.shape)


    # train_loader = datasets.fetch_dataloader(args)

    # backbone = RAFTAvgBackbone(args)
    # backbone.eval()
    # for i_batch, data_blob in enumerate(train_loader):
    #     image1, image2, flow, valid = [x.cuda() for x in data_blob]
    #     if args.add_noise:
    #         stdv = np.random.uniform(0.0, 5.0)
    #         image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
    #         image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
    #     print(image1.dtype)
    #     print("#####", image1.shape, image2.shape, flow.shape)

    #     features = backbone(image1, image2, iters=args.iters)
    #     print(features.shape)
    #     break

    # backbone = RAFTTransformerBackbone(args)
    # backbone.eval()
    # for i_batch, data_blob in enumerate(train_loader):
    #     image1, image2, flow, valid = [x.cuda() for x in data_blob]
    #     if args.add_noise:
    #         stdv = np.random.uniform(0.0, 5.0)
    #         image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
    #         image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
    #     print("#####", image1.shape, image2.shape, flow.shape)

    #     features = backbone(image1, image2, iters=args.iters)
    #     print(features.shape)
    #     break