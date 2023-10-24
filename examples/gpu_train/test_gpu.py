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


import argparse
import os
import sys
import cv2
import shutil
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
from collections import OrderedDict
from PIL import Image
import PIL
import timm
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
import torchpipe


class InferGPUDecode():
    def __init__(self, args):
        self.toml_path = args.toml_path
        self.arch = args.arch
        self.num_classes = args.num_classes
        self.model_path = args.model_path
        self.class_dict = {i: item for i, item in enumerate(
            args.class_label.strip().split(','))}
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.decode_node = self.init_decodeNode()
        self.model = self.init_model()
        pass

    # 初始化torchpipe
    def init_decodeNode(self):
        config = torchpipe.parse_toml(self.toml_path)
        for key in config.keys():
            if key != 'global':
                # toml里面没有指定gpu，这里指定为0
                config[key]["device_id"] = 0
        print(config)
        decode_node = torchpipe.pipe(config)
        return decode_node

    def load_param(self, model, pretrained_model):
        state_dict = torch.load(pretrained_model)
        if "state_dict" in state_dict:
            state_dict = state_dict.get("state_dict")
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if not "tracked" in key:
                key_name = key
                if "module" in key:
                    key_name = key.split(".", 1)[1]
                if value is None:
                    print(
                        "Value of key {} is None in pretrained model.".format(key_name))
                    continue
                if model.state_dict().get(key_name) is None:
                    print("Value of key {} is None in model.".format(key_name))
                    continue
                if value.shape != model.state_dict().get(key_name).shape:
                    print("layer {} skip, for shape can't match:"
                          "{}(model) vs {}(pretrained)".format(
                              key_name, str(model.state_dict().get(key_name).shape, str(value.shape))))
                    continue
                new_state_dict[key_name] = value

        model.load_state_dict(new_state_dict, strict=False)
        return model

    # 初始化pytorch model

    def init_model(self):
        if self.arch == 'resnet50':
            model = timm.create_model('resnet50', pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            model = self.load_param(model, self.model_path)

        model.cuda().eval()
        return model

    # 方法一：
    # infer function
    # 解码和模型，全部使用torchpipe，详见gpu_decode_test.toml

    def predict_1(self, img_path):

        try:
            img = open(img_path, 'rb').read()
            if img is None or img == b'':
                print('open error:{}'.format(img_path))
                return None

            pipe_input = {TASK_DATA_KEY: img, "node_name": "jpg_decoder"}
            self.decode_node(pipe_input)
            result = pipe_input[TASK_RESULT_KEY]
            result = torch.nn.functional.softmax(result).cpu().numpy()
            return result

        except Exception as e:
            print('error:{}'.format(e))
            return None

    # 方法二：
    # infer function
    # 解码使用torchpipe gpu，模型前向不使用torchpipe,使用pytorch的原生模型
    def predict_2(self, img_path):

        try:

            img = open(img_path, 'rb').read()
            if img is None or img == b'':
                print('open error:{}'.format(img_path))
                return None

            pipe_input = {TASK_DATA_KEY: img, "node_name": "jpg_decoder"}
            self.decode_node(pipe_input)
            img_data = pipe_input[TASK_RESULT_KEY]
            mean_tensor = torch.tensor(self.mean).unsqueeze(0).unsqueeze(
                2).unsqueeze(3).repeat(1, 1, 224, 224).cuda()
            std_tensor = torch.tensor(self.std).unsqueeze(0).unsqueeze(
                2).unsqueeze(3).repeat(1, 1, 224, 224).cuda()
            img_data -= mean_tensor
            img_data /= std_tensor

            with torch.no_grad():
                probs = self.model(img_data)
                result = torch.nn.functional.softmax(probs).cpu().numpy()
            return result

        except Exception as e:
            print('error:{}'.format(e))
            return None

    def infer(self, base_path, count, result_path):

        print('process:{}'.format(base_path))

        files = os.listdir(base_path)
        for file in files:

            img_path = os.path.join(base_path, file)

            try:
                # method 1
                # cla = self.predict_1(img_path)
                # method 2
                cla = self.predict_2(img_path)
                if cla is None:
                    continue
            except Exception as e:
                print('predict error:{}'.format(e))
                continue

            max = np.argmax(cla[0])

            result_name = 'cla'
            for i in range(self.num_classes):
                result_name += '-{}-{:.4f}'.format(
                    self.class_dict[i], cla[0][i])

            result_name += ('-' + file)

            save_result_path = os.path.join(result_path, self.class_dict[max])

            if not os.path.exists(save_result_path):
                os.makedirs(save_result_path)
            try:
                shutil.copyfile(img_path, os.path.join(
                    save_result_path, result_name))
            except:
                print('shutil error')
            count += 1

            if count % 100 == 0:
                print('--{}--'.format(count))

    def main(self, args):

        base_path = args.test_images_path
        if not os.path.exists(base_path):
            print('images file path error')
            return

        result_path = args.test_result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        count = 0

        self.infer(base_path, count, result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--toml-path', type=str,
                        default='./toml/gpu_decode_val.toml')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--class-label', type=str, default='porno,sexy,normal')
    parser.add_argument('--test-images-path', type=str, default='None')
    parser.add_argument('--test-result-path', type=str, default='None')
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    InferClient = InferGPUDecode(args)
    InferClient.main(args)
