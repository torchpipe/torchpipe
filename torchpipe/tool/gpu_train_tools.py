# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 将torchpipe引入到pytorch训练中，从训练端对齐与部署时候采用gpu解码带来的差异，提高部署性能。
# 支持gpu与cpu同时解码


import torch
import os

import torchpipe 

from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY

import threading
import concurrent
import queue
import random
import cv2
import numpy as np

class Dataloader:

    def __init__(self, parallel_type, dataloader, toml_path, transforms_gpu=None, transforms_cpu=None, cpu_percentage=0.5, local_rank=None, devices=None):
        """
        初始化方法,用于创建torchpipe的训练dataloader迭代器。
        :param parallel_type:  required, 采用的并行训练的方法,可选两个参数: dp和ddp
        :param dataloader:     required, 使用的dataloader即可
        :param toml_path:      required, torchpipe的用于解码的toml_path
        :param transforms_gpu: option,   当需要使用gpu解码的时候,需要这个选项
        :param transforms_cpu: option,   当需要cpu解码的时候,需要这个选项
        :param cpu_percentage: option,   cpu解码的比例,0表示不使用cpu解码,1 表示全部使用cpu解码,0-1之间为比例
        :param local_rank:     option,   当使用ddp的时候,必须指定
        :param devices:        option,   仅dp时候生效,默认使用本级可训练的全部显卡训练,如果只使用部分显卡,需要使用该参数。
        """
        
        self.parallel_type = parallel_type
        self.dataloader = dataloader
        self.transforms_gpu = transforms_gpu
        self.transforms_cpu = transforms_cpu
        self.cpu_percentage = cpu_percentage
        self.local_rank = local_rank
        self.device_ids  = None if devices == None else [int(item) for item in devices.strip().split(',')]
        
        ## for torchpipe 
        self.toml_path = toml_path
        self.node = self.__init_nodes__()

        ## multi thread 
        self.queue = queue.Queue(maxsize=4)
        self.executor= concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.thread_start = False
        self.stop = False
        self.executor_cpu_aug = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.__init_param_check__()

    def __init_param_check__(self):

        assert(self.cpu_percentage <= 1 and self.cpu_percentage >= 0) , 'cpu percentage must in 0 -1 '

        assert(self.parallel_type == 'dp' or self.parallel_type == 'ddp') , 'parallel_type must dp or ddp'

        if self.parallel_type == 'ddp' and self.local_rank is None:
            print('error: ddp mode must set local rank')

        if self.transforms_gpu is None and self.transforms_cpu is None:
            print('error: transforms_cpu and transforms_gpu are all None !!!')
            exit(0)

        if self.transforms_cpu is None and self.cpu_percentage != 0:
            print('error: transforms_cpu is None, cpu percentage must be set to 0 !!!')
            exit(0)
        if self.transforms_gpu is None and self.cpu_percentage != 1:
            print('error: transforms_gpu is None, cpu percentage must be set to 1 !!!')
            exit(0)
        


    
    def __init_nodes__(self):
        if self.parallel_type == 'dp':
            nodes = []
            config = torchpipe.parse_toml(self.toml_path)
            if self.device_ids is None:
                # 如果没有指定device_ids, 则默认使用所有可用的gpu
                if torch.cuda.is_available():
                    self.device_ids = [ index for index in range(torch.cuda.device_count())]
        
            for i in self.device_ids :
                for key in config.keys():
                    if key != 'global':
                        config[key]["device_id"] = str(i)
                nodes.append(torchpipe.pipe(config))
            return nodes

        elif self.parallel_type == 'ddp':
            config = torchpipe.parse_toml(self.toml_path) 
            for key in config.keys():
                if key != 'global':
                    if self.local_rank is not None:
                        config[key]["device_id"] = str(self.local_rank)
                    else:
                        print('Error: local rank param error')
                        exit(0)
            print(config)
            node = torchpipe.pipe(config)
            return node


    def __iter__(self):

        if self.thread_start == False:
            self.thread = threading.Thread(target=self.__generator__)
            self.thread.start()
            self.thread_start = True

        while True:
            if self.stop==True and self.queue.empty():
                break
            try:
                data = self.queue.get(timeout=5)
            except Exception as e:
                print("error inner: {}".format(e))   
                continue

            try:
                data=data.result()
                if data is None:
                    print('error occured when torchpipe decode, frame has ignored it ')
                    continue

                yield data

            except Exception as e:
                print("wrap loader error inner:{}".format(e))   
                continue



    def __generator__(self):
       
        for data in self.dataloader:
            if len(data) != 1:
                params = (data[0], data[1:])
            else:
                params = (data, None)

            if random.random() < self.cpu_percentage:
                future = self.executor.submit(self.decoder_cpu_and_aug, params)
            else:
                future = self.executor.submit(self.decoder_gpu_and_aug, params)

            self.queue.put(future)

        self.stop = True


    def decoder_gpu_and_aug(self, params):
        input_bytes, target = params
        input_tensor = []
        input_dicts = []
        try:
            for index in range(len(input_bytes)):
                input_dict = {TASK_DATA_KEY: input_bytes[index], "node_name": "jpg_decoder"}
                input_dicts.append(input_dict)
            if self.parallel_type == 'ddp':
                self.node(input_dicts)
            elif self.parallel_type == 'dp':
                if len(self.device_ids) > 1:
                    now_id = random.choice(self.device_ids)
                self.node[now_id](input_dicts)

            for item in input_dicts:    
                input_tensor.append(item[TASK_RESULT_KEY])
            
            input = torch.cat(input_tensor, dim=0)
            input = input.to(torch.uint8)  
            input = self.transforms_gpu(input)
        except Exception as e:
            print('torchpipe decode error')
            return None

        if target is not None:
            result = input, *target
        else:
            result = input
        return result
    
    def decoder_gpu_and_aug2(self, params):
        """
        该方法是作为上面gpu解码的补充，与上面方法的区别在于，上面是一个batch过torchpipe，但是如果解码出错，一个batch都会丢掉
        这个方法，采用一张一张过，如果解码失败，只会丢掉那一张图像，提高了图像的利用率，但是因为增加了许多的额外的对齐以及check操作
        所以，整体效率，比上面方法要低。
        """
        input_bytes, target = params

        if target is not None:
            target_result = [[] * len(target)] 
        input_tensor = []

        for index in range(len(input_bytes)):
            try:
                input_dict = {TASK_DATA_KEY: input_bytes[index], "node_name": "jpg_decoder"}

                if self.parallel_type == 'ddp':
                    self.node(input_dict)
                elif self.parallel_type == 'dp':
                    self.node[self.device_ids[0]](input_dict) ## 只在一块卡上解码，因为两块卡没法torch.cat


                input = input_dict[TASK_RESULT_KEY]
                input = input.to(torch.uint8)  
                input = self.transforms_gpu(input)

                input_tensor.append(input)

                if target is not None:
                    for target_i in range(len(target)):
                        if isinstance(target[target_i], torch.Tensor):
                            # 如果是tensor，需要保持维度不变，所以需要加unsqueeze
                            target_result[target_i].append(target[target_i][index].unsqueeze(0))
                        else:
                            target_result[target_i].append(target[target_i][index])

            except Exception as e:
                print('torchpipe decode error')
                continue

        input_tensor = torch.cat(input_tensor, dim=0)

        for i in range(len(target)):
            if isinstance(target[i], torch.Tensor):
                target_result[i] = torch.cat(target_result[i], dim=0)


        ## check result， must have same batchsize
        if target is not None:
            input_batch = input_tensor.shape[0]
            all_same = True
            for i in range(len(target_result)):
                if isinstance(target_result, torch.Tensor):
                    if target_result[i].shape[0] != input_batch:
                        all_same = False
                        break
                elif isinstance(target_result, list):
                    if len(target_result[i]) != input_batch:
                        all_same = False
                        break
                else:
                    print('not support this data type')
                    all_same = False
                    break
            
            if all_same == False:
                return None
        
        if target is not None:
            result = input_tensor, *target_result
        else:
            result = input_tensor
        return result

    
    def decoder_cpu_and_aug(self, params):
        input_bytes, target = params
        inputs = []
        try:
            task_iter = self.executor_cpu_aug.map(self.cpu_augment, input_bytes)
            for task in task_iter:
                inputs.append(task)
            inputs = torch.cat(inputs, dim=0)
        except Exception as e:
            print('error: cpu decode error')
            return None
        
        if target is not None:
            result = inputs, *target
        else:
            result = inputs

        return result


    def cpu_augment(self, input_byte):
        input_data = np.frombuffer(input_byte, dtype="uint8")
        input_data = cv2.imdecode(input_data, flags=cv2.IMREAD_COLOR)
        input_data = self.transforms_cpu(input_data)
        input_data = input_data.unsqueeze(0)
        return input_data

    def reset(self):
        self.thread_start = False
        self.stop = False

    def __len__(self):
        return len(self.dataloader)



class TensorToTensor(torch.nn.Module):
    """change tensor dtype to torch.float32, and change from [0,255] -> [0, 1].
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        if not img.dtype == torch.float32:
            return img.to(torch.float32) / 255.0
        return img / 255.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


def cv2_loader(path):
    img_bytes =  open(path,'rb').read()
    return img_bytes


