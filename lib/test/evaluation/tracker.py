import torch
import importlib
import os
import torch.nn as nn
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import clip
import numpy as np
from PIL import Image
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from transformers import BertTokenizer, BertModel
import sys
from torchvision.utils import save_image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator


device = "cuda" if torch.cuda.is_available() else "cpu"
#clip_model, clip_transform = clip.load("RN101", device=device)  # use time
#clip_model = clip_model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = '/data1/xuelin/xuelin/stark5/bert-base-uncased/'
bert_model = BertModel.from_pretrained(model_path).to(device)
bert_model.eval()
def tensor2im(input_image, imtype=np.uint8):
        """"
        Parameters:
            input_image (tensor) 
            imtype (type)        
        """
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]  
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor): 
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  
            if image_numpy.shape[0] == 1:  
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)): 
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255 
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  
        else: 
            image_numpy = input_image
        return image_numpy.astype(imtype)

def img2seq(img):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    #ckpt_dir = "/data1/xuelin/xuelin/Stark/ofa-large-caption/"
    #tokenizer = OFATokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    
    ckpt_dir = "/data1/xuelin/xuelin/Stark/ofa-large-caption/"
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
    
    #model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
    ofa_model = OFAModel.from_pretrained(ckpt_dir)
    ofa_model.eval
    generator = sequence_generator.SequenceGenerator(
        tokenizer=tokenizer,
        beam_size=5,
        max_len_b=16,
        min_len=0,
        no_repeat_ngram_size=3,
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ofa_model = ofa_model.to(device)
    generator = generator.to(device)
    
    patch_img = patch_resize_transform(img).unsqueeze(0)
    
    patch_img = patch_img.to(device)
    inputs = tokenizer(["what does the image describe?"], return_tensors="pt").input_ids.to(device)
    data = {}
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks': torch.tensor([True]).to(device)}
    gen_output = generator.generate([ofa_model], data)
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
    
    gen = ofa_model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
    # print(tokenizer.batch_decode(gen, skip_special_tokens=True))
    
    return tokenizer.batch_decode(gen, skip_special_tokens=True)
def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        #caption = param_list[i][0].name.split("-")[0]
        
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()  #info with caption

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
    
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        #print("tracker:",tracker)
        #print("tracker.network",tracker.network)
        #print("tracker.network.linear",tracker.network.linear)
        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
        
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        
        image = self._read_image(seq.frames[0])
        caption = init_info['caption']
        experience = []
        sequence_name = caption
        output_path = '/data1/xuelin/xuelin/stark6/test/lasot'
        height, width, channels = image.shape  
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = 30
        output_video_path = os.path.join(output_path,f'{caption}.mp4')
        output_video = cv.VideoWriter(output_video_path, fourcc, fps,(width,height))

        '''
				caption = clip.tokenize(caption).to(device)
        text_feature = clip_model.encode_text(caption).float()
        text_feature = linear(text_feature)
        text_feature = tracker.linear(text_feature)
        '''
        #print("type of image",type(image)) #numpy
        pillow_image = Image.fromarray((np.array(image, dtype=np.uint8)))
        print("type of pillow_image",type(pillow_image))
        if caption is None:
            caption = img2seq(pillow_image)
        with torch.no_grad():
            print("caption:",caption)
            caption = tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = bert_model(**caption)
            text_feature = outputs.last_hidden_state.mean(dim=1).float()
            text_feature = tracker.network.linear(text_feature)
        #print("shape of text_feature",text_feature.shape) #[1, 256]

        start_time = time.time()
        out = tracker.initialize(image, init_info,text_feature)
        
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'caption':init_info.get('caption')}
        
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            
            info['previous_output'] = prev_output
            out = tracker.track(experience,text_feature, image, info)
            img = cv.imread(frame_path)
            pred_bbox = out['target_bbox']
            save_img_path = os.path.join(output_path, f'{sequence_name}_{frame_num}.jpg') 
            cv.rectangle(img,(int(pred_bbox[0]),int(pred_bbox[1])),(int(pred_bbox[0]+pred_bbox[2]),int(pred_bbox[1]+pred_bbox[3])),(0,0,255),3)
            #cv.imshow(seq.name, img)
            #cv.waitKey(1) # 0or1??
            
            #output_video.write(img)
             
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        #output_video.release()
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



