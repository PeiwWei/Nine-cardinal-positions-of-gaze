import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from model.ICSA.transformer_block import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerHELayer
)
from collections import OrderedDict
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding　
    这里的一系列操作都是为了实现patch的序列化
    [16（块大小） * 3（3张） * 4 * 3(比例), 16（块大小） * 3（3张） * 4]
    embed_dim = 16 * 16 * 3
    输出为[B ,num_patches,  embed_dim] = [B , 36 * 12 , 768]
    """
    def __init__(self, img_size=[192, 576], patch_size=[16, 16], in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size[0]
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # patches 的数量， 卷积核的大小和分割的patches的大小相同都是16 * 16
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # 若没有传入norm_layer则对其不进行任何操作，直接进入Identity
    def forward(self, x):
        # VIT模型输入的大小必须是固定的
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def ICSA( num_classes: int = 10, has_logits: bool = True):
    '''
    args:
    img_size (int, tuple): input image size
    depth (int): depth of transformer
    nheads (int): number of attention heads
    block_size(int, tuple): num of all_blocks
    mlp_dim (int): dim of mlp
    embed_dim (int): embedding dimension ( h_b * w_b )

    '''
    model = ICSA_TR(img_size = [192, 576],
                    dim=768, depth=6, heads=12,
                    mlp_dim=256, pe_dim=(192, 576),
                    num_classes = 3, embed_dim = 768,
                    has_logits = True)

    return model



class ICSA_TR(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, pe_dim,
                 img_size = [192, 576], patch_size=[16, 16], in_c=3, block_size =[4, 12],
                 num_classes = 3, embed_dim =768, drop_ratio=0.,
                 embed_layer = PatchEmbed,
                 norm_layer=None, act_layer=None,
                 has_logits = True):
        # 这里的block_size 是 分成块的大小
        """
        num_tokens : cls
        drop_ratio (float): dropout rate
        [B , C , H ,W] --> [B, num_patches, embed_dim]
        embed_dim = 16* 16 *3
        """
        super(ICSA_TR,self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.num_tokens = 1

        self.patch_embed = embed_layer(img_size=img_size, patch_size = patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)


        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm = norm_layer(embed_dim)

        enc_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, block_size=block_size, block_num = [3, 3])
        self.encoder = TransformerEncoder(enc_layer, num_layers=depth)

        self.pe_enc = nn.Parameter(torch.rand([1, dim, pe_dim[0], pe_dim[1]]), requires_grad=True)

        self.has_logits = False
        # Classifier head(s)
        self.head1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.pre_logits = nn.Identity()

        # 定义全局平均池化层
        self.global_pooling = nn.AdaptiveAvgPool2d((1, None))

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_tr_weights)


            
    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 432, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)  # [B, 433, 768]
        # x = self.pos_drop(x + self.pos_embed)

        x = self.encoder(x, pos=self.pos_embed)
        x = self.norm(x)
        # x = self.pre_logits(x[:, 0])
        x = self.global_pooling(x).view(x.size(0), -1)
        output1 = self.head1(x)
        output2 = self.head2(x)



        return output1, output2



def _init_tr_weights(m):
    """
    tr weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

if __name__ == '__main__':
    import random
    import numpy as np

    model = ICSA_TR(dim=512, depth=1, heads=2, mlp_dim=256, pe_dim=(128, 256))
    model.cuda().eval()

    x_enc = torch.rand(1, 512, 1, 128, 256).cuda()
#     with torch.no_grad():
    while True:
        preds = model(x_enc, x_enc, x_enc)
        print(preds.shape)
    
#     model = ERM(dim=512, mlp_dim=256, KL=10, KH=10, num_class=19)
#     model.cuda().eval()

#     x = torch.rand(1, 512, 128, 256).cuda()
#     with torch.no_grad():
#         preds = model(x)
#         print(preds.shape)