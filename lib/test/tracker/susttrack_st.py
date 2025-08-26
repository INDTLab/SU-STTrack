from lib.test.tracker.basetracker import BaseTracker
import torch
from matplotlib import pyplot as plt
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import random
import os
from lib.utils.merge import merge_template_search
from lib.models.susttrack import build_starkst
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
import sys

def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


class STARK_ST(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_ST, self).__init__(params)
        network = build_starkst(params.cfg)
        print("self.params.checkpoint",self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

    def initialize(self, image, info: dict,text_feature):
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
            text_feature = torch.unsqueeze(text_feature,dim=0).expand(self.z_dict1['feat'].shape[0],-1,-1)
            self.z_dict1['feat'] = self.z_dict1['feat'] + text_feature
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, experience, text_feature, image, info: dict = None):
        # print("shape of text_feature in track",text_feature.shape) #[1, 256]
        H, W, _ = image.shape
        initial_text_feature = text_feature
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        #print("---------------------------------")
        plt.figure(figsize=(10, 10))
        plt.imshow(x_patch_arr)  
        plt.title("Original Image")
        plt.axis('off')  
        plt.savefig("original_image.png", dpi=300) 
        plt.show()                                                       
                                                                        
        #print("66666666666666666666666666666666666666666")                                                        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
       # print("type of x_patch_arr",type(x_patch_arr))
        #print("shape of x_patch_arr",x_patch_arr.shape) (320,320,3)
       
        with torch.no_grad():
            
            x_dict = self.network.forward_backbone(search)
            #print("x_dict",x_dict)
            #sys.exit()
            
            

            text_feature = torch.unsqueeze(text_feature,dim=0).expand(x_dict['feat'].shape[0],-1,-1)
            
            x_dict['feat'] = text_feature + x_dict['feat']  #bad!!
            prompt_feature = x_dict['feat'].cpu()
            prompt_feature = prompt_feature.squeeze(1).view(20,20,256)
            mean_prompt_feature = torch.mean(prompt_feature,dim=2)
            plt.figure(figsize=(10, 10))
            plt.imshow(mean_prompt_feature.numpy(), cmap='viridis')
            plt.colorbar()  
            plt.title("Fused Prompt Feature Visualization")
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.tight_layout()
            plt.savefig("mean_prompt_feature_map.png", dpi=300)
            plt.show()
            #print("shape of x_dict['feat']",x_dict['feat'].shape)  #400,1,256
            
            
            # merge the template and the search
            
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            #print("seq_dict",seq_dict)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # get experience feat
        if self.frame_id % 100 == 0 and conf_score > 0.6:
            temp_patch_arr, _, temp_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
            temp_template = self.preprocessor.process(temp_patch_arr, temp_amask_arr)
            with torch.no_grad():
                temp_dict = self.network.forward_backbone(temp_template)
                text_feature = torch.unsqueeze(initial_text_feature,dim=0).expand(temp_dict['feat'].shape[0],-1,-1)
                temp_dict['feat'] = text_feature + temp_dict['feat'] 
            experience.append(temp_dict['feat'])
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                    text_feature = torch.unsqueeze(initial_text_feature,dim=0).expand(z_dict_t['feat'].shape[0],-1,-1)
                    z_dict_t['feat'] = text_feature + z_dict_t['feat']
                #self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
                #print("shape of self.z_dict_list[idx+1]",self.z_dict_list[idx+1]['feat'].shape)  [64, 1, 256]
                #sys.exit()
                #self.z_dict_list[idx+1] = z_dict_t
                # experience replay
                if len(experience) > 3:
                    rand_experience = random.choice(experience)
                    self.z_dict_list[idx+1]['feat'] =0.6*z_dict_t['feat'] +0.4*rand_experience
                    #print("shape of self.z_dict_list[idx+1]",self.z_dict_list[idx+1]['feat'].shape) [64, 1, 256]

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_ST
