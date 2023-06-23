import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_


class ViT_CoTrain(VisionTransformer):

    def __init__(self, *args, num_labels, lost_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.lost_rate = lost_rate
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_labels)
        )

        trunc_normal_(self.pos_embed, std=.02)


    def forward_features(self, x, test=0):

        B = x.shape[0]
        #print(f"Before Patch_Embed:{x.shape}")
        x = self.patch_embed(x)
        #print(f"After Patch_Embem:{x.shape}")
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        #print(cls_tokens.shape)
        x = torch.cat((cls_tokens,  x), dim=1)
        #print(x.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if not test:
            for blk in self.blocks:
                #permutation is a tensor
                permutation = torch.randperm(B).to('cuda')
                #print(permutation)
                temporal = x[permutation][:int(B*(1-self.lost_rate))]
                t = blk(temporal)
                t = (1/(1-self.lost_rate))*t
                x.index_add_(0,permutation[:int(B*(1-self.lost_rate))],t)
                #print(f"AFTER ATTN:{x.shape}")

            x = self.norm(x)
            return x[:,0]
        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            return x[:,0]
        
    def forward(self, x, test=0):

        cls_tokens = self.forward_features(x, test)
        return self.ffn(cls_tokens)

