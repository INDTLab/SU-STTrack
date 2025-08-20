import torch
import sys
import os

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = '/data1/xuelin/xuelin/stark5/bert-base-uncased/'
model = BertModel.from_pretrained(model_path).to(device)
model.eval()
'''
dataset_path = "/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/lasot/"
def iterate_text_files(dataset_path):
    for dirs in os.listdir(dataset_path):
        sub_dir_path = os.path.join(dataset_path,dirs)
        for final_dir in os.listdir(sub_dir_path):
            final_dir_path = os.path.join(sub_dir_path,final_dir)
            print("final_dir_path",final_dir_path)
            contents = os.listdir(final_dir_path)
            if 'nlp.txt' in contents:
                nlp_path = os.path.join(final_dir_path, 'nlp.txt')
                if os.path.getsize(nlp_path) == 0:
                    print("File is empty, stopping iteration.")
                    return
            else:
                print("no nlp file")
                print("final_dir_path",final_dir_path)


iterate_text_files(dataset_path)
