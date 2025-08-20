"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import matplotlib.pyplot as plt
import random


class STARKS(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.linear = nn.Linear(768,256)
        
        
        #channel attention
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1,stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  #for layer2

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)  
        self.sigmoid = nn.Sigmoid()
        
        
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos, low_feat = self.backbone(input)  # features & masks, position embedding for the search
        #output_back, pos = self.backbone(input)
        
      
        
        high_feature = output_back[0].tensors  # [16, 1024, 8, 8]
        
        mean_high_feature = torch.mean(high_feature, dim=1)  # [16, 8, 8]
        
        sample_idx = 0
        sample_mean_high = mean_high_feature[sample_idx]  # [8, 8]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(sample_mean_high.cpu().squeeze().numpy(), cmap='viridis')
        plt.title("Average High Feature")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("average_high_feature_map.png", dpi=300)
        plt.show()
       
        low_feature = low_feat['0']  # [1, 512, 16, 16]
        
     
        mean_low_feature = torch.mean(low_feature, dim=1)  # [1, 16, 16]
        
        sample_idx = 0
        sample_mean_low = mean_low_feature[sample_idx]  # [16, 16]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(sample_mean_low.cpu().squeeze().numpy(), cmap='viridis')
        plt.title("Average Low Feature")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("average_low_feature_map.png", dpi=300)
        plt.show()
        
       
        low_feature = self.conv1(low_feature)
        low_feature = self.avg_pool(low_feature)  # [16, 1024, 8, 8]
        cat_feature = torch.cat((low_feature, output_back[0].tensors), dim=1)  # [16, 2048, 8, 8]
        
        # channel attention
        cat_feature = self.global_pool(cat_feature)  # [16, 1024, 1, 1]
        cat_feature = self.conv2(cat_feature)        # [16, 1024, 1, 1]
        cat_feature = self.relu(cat_feature)         # [16, 1024, 1, 1]
        cat_feature = self.conv3(cat_feature)        # [16, 1024, 1, 1]
        weight = self.sigmoid(cat_feature)           # [16, 1024, 1, 1]
        low_feature = weight * low_feature
        final = low_feature + high_feature            # [16, 1024, 8, 8]
        
       
        mean_fusion_feature = torch.mean(final, dim=1)  # [16, 8, 8]
        
        sample_idx = 0
        sample_mean_fusion = mean_fusion_feature[sample_idx]  # [8, 8]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(sample_mean_fusion.cpu().squeeze().numpy(), cmap='viridis')
        plt.title("Average Fusion Feature")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("average_fusion_feature_map.png", dpi=300)
        plt.show()
        
        
        output_back[0].tensors = final
        
        
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                 seq_dict["pos"], return_encoder_output=True)
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, enc_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)

        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        #print(feat_vec.shape)
        '''
        temp = feat_vec
        temp = temp.cpu()
        if temp.shape[0] == 64:
            temp = temp.squeeze(1).view(8,8,256)
        if temp.shape[0]==400:
            temp = temp.squeeze(1).view(20,20,256)    
        mean_prompt_feature = torch.mean(temp,dim=2)
        plt.figure(figsize=(10, 10))
        plt.imshow(mean_prompt_feature.numpy(), cmap='viridis')
        plt.colorbar()  
        plt.title("Test in Adjuct")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig("adjust.png", dpi=300)
        plt.show()
        '''
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_starks(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    model = STARKS(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model
