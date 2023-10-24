# from __future__ import print_function
import numpy as np
from torchvision import models,transforms
from timeit import default_timer as timer
from serve.ttypes import InferenceResult
import torchvision.models as models
import threading

from torch2trt import torch2trt
import torch
import cv2


class RecognizerHandler():
    def __init__(self, args=None):
        self.cls_trt_max_batchsize = 1
        self.resize_size = 320 
        self.fp16 = True
        self.lock = threading.Lock()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().cuda()
        input_shape = torch.ones((1, 3, 224, 224)).cuda()
        self.classification_engine = torch2trt(resnet50, [input_shape], 
                                                fp16_mode=self.fp16,
                                                max_batch_size=self.cls_trt_max_batchsize,
                                                )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.pre_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225]),])
        print("finish init")



    def preprocess(self, img):
        orgimages_list = []
        orgimages_list.append(self.pre_trans(cv2.resize(img[:, :, (2, 1, 0)], (224, 224))))  
        bin_data = torch.stack(orgimages_list, axis=0).cuda()
        return bin_data

    
    def infer_batch(self, bin_list):
        results = InferenceResult()
        img = cv2.imdecode(np.asarray(bytearray(bin_list.data), dtype='uint8'), flags=cv2.IMREAD_COLOR) #bgr
        bin_data = self.preprocess(img)
        try:
            with self.lock:
                result_tmp = self.classification_engine(bin_data)

            dis = self.softmax(result_tmp)                      
            maxprob,label_index = torch.max(dis,1)
            label_index_tmp = label_index.item()
            maxprob_tmp = maxprob.item()
            
            results.label = str(label_index_tmp)
            results.score = float(maxprob_tmp)
            return results

        except Exception as e:
            print("uuid={:} : {}".format(bin_list.uuid, e))
            return results

    def ping(self):
        print('ping()')

