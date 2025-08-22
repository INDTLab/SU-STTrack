from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from lib.utils.merge import merge_template_search
from PIL import Image
import numpy as np
import os
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import sys
import clip
from transformers import BertTokenizer, BertModel

device = "cuda" if torch.cuda.is_available() else "cpu"
#clip_model, clip_transform = clip.load("RN101", device=device)
#clip_model = clip_model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = '/data1/xuelin/xuelin/stark6/bert-base-uncased/'
model = BertModel.from_pretrained(model_path).to(device)
model.eval()
class STARKSActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox' and 'test_class'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            # captions = data['test_class'] # for class name
            # captions = data['nlp'] # for fine caption
            captions = data['exp_str']
            '''
            templatecpu = template_img_i.cpu()
            templatenumpy = templatecpu.numpy()
            for j in range(templatenumpy.shape[0]):
                caption = captions[j]
                plt.imshow(templatenumpy[j].transpose(1, 2, 0))  # Transpose to (128, 128, 3) for imshow
                plt.title(caption)
                plt.show()
                plt.pause(1)
            sys.exit()
            '''
            feature_dict = self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone')
            image_features = feature_dict['feat']  # [64, 16, 256]
            text_features = self.text_bert_net(self.net, captions)
            text_features = torch.stack(text_features, dim=1)
            # print("shape of text features",text_features.shape)  # [1, 16, 256]
            text_features = torch.squeeze(text_features)
            text_features = torch.unsqueeze(text_features,dim=0).expand(64,-1,-1) 
            # feature = torch.cat((image_features, text_features), dim=-1)

            feature = image_features + text_features
            # print("shape of feature",feature.shape) # [64, 16, 256]
            feature_dict['feat'] = feature
            feat_dict_list.append(feature_dict)
            #feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone'))

        # process the search regions (t-th frame) 
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        #captions = data['test_class']
        '''
        search_img_cpu = search_img.cpu()
        print("shape of search_img_cpu",search_img_cpu.shape)
        save_path = '/data1/xuelin/xuelin/stark5/lib/train/actors/'
        for j in range(search_img_cpu.shape[0]):
            caption = captions[j]
            file_path = os.path.join(save_path, f'img_{caption}.png')
            save_image(search_img_cpu[j], file_path)
        sys.exit()
        '''
        feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))
        '''
        feature_dict = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
        image_features = feature_dict['feat']
        # print("shape of image_features",image_features.shape)  #[64, 16, 256]
        text_features = self.text_bert_net(self.net, captions)
        text_features = torch.stack(text_features, dim=1)
        # print("shape of text_features",text_features.shape)  #[1, 16, 256]
        text_features = text_features.expand(image_features.shape[0],-1,-1)
        feature = image_features + text_features
        feature_dict['feat'] = feature
        feat_dict_list.append(feature_dict)
        '''
        
        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list)
        #print("type of seq_dict['feat']",type(seq_dict['feat'])) #tensor
        #print("shape of seq_dict['feat']",seq_dict['feat'].shape) #[528, 16, 256] 
        out_dict, _, _ = self.net(seq_dict=seq_dict, mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    '''
    def text_clip_net(self, net, captions):
        # input of clip model should be tensor, instead of str
        text_features = []
        for caption in captions:
            caption = clip.tokenize(caption).to(device)
            text_feature = clip_model.encode_text(caption).float()
            text_feature = net.module.linear(text_feature)
            #print("text_feature",text_feature)
            #print("shape of text_feature",text_feature.shape)
            text_features.append(text_feature)
        return text_features
    '''
    
    def text_bert_net(self,net,captions):
          text_features=[]
          for caption in captions:
              caption = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
              outputs = model(**caption)
              text_feature = outputs.last_hidden_state.mean(dim=1).float()
              text_feature = net.module.linear(text_feature)
              text_features.append(text_feature)
          return text_features
    
    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
