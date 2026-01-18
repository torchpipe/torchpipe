# ACL Inference Script for Huawei NPU
# Usage：
'''
from acl_helper import OmModel
model = OmModel('model/classifier.om')
inputs = torch.randn(4, 3, 224, 224)
outputs = model(inputs)
print(outputs.shape)
# For models with dynamic batch sizes, by default, padding to the nearest batch size is used for inference. You can disable padding and use the split method for inference (e.g., input 5x3x224x224 can be split into [4x3x224x224, 1x3x224x224]).
model = OmModel('model/classifier_dynamic.om', padding_batch=False)
'''
# Thread Safety
'''
from acl_helper import OmModel, ThreadSafeWrapper
om = OmModel("model/detector.om")
om = ThreadSafeWrapper(om)
'''
# For the latest version and further usage instructions, please refer to:  https://g.hz.netease.com/deploy/helper

# 20241205: 修复输入数据在cpu上无法正确处理的问题

from timeit import default_timer as timer
import threading
from typing import Any, Union, List
import acl
from abc import ABCMeta, abstractmethod
import subprocess
import os
import torch_npu
import torch

# import threading


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}".format(message, ret))


def device_guard():
    device_id, ret = acl.rt.get_device()
    if (ret != 0):

        count, ret = acl.rt.get_device_count()
        check_ret("acl.rt.get_device_count", ret)
        if count == 0:
            raise Exception("No device found")
        else:
            print("Found {} devices".format(count))

        ret = acl.rt.set_device(0)
        check_ret("acl.rt.set_device(0)", ret)

    return device_id


def init_acl():
    torch.npu.set_device(torch.device("npu:0"))
    device_guard()
    # managed by pytorch
    if False:
        ret = acl.init()
        check_ret("acl.init", ret)
        print("acl.init success")


_ = init_acl()


def destroy_data_set_buffer(dataset):

    num = acl.mdl.get_dataset_num_buffers(dataset)
    for i in range(num):
        data_buf = acl.mdl.get_dataset_buffer(dataset, i)
        if data_buf:
            ret = acl.destroy_data_buffer(data_buf)
            check_ret("acl.destroy_data_buffer", ret)
    ret = acl.mdl.destroy_dataset(dataset)
    check_ret("acl.mdl.destroy_dataset", ret)


class ThreadSafeWrapper:
    def __init__(self, om_model):
        self.om_model = om_model
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            return self.om_model(*args, **kwargs)


class OmModel(metaclass=ABCMeta):
    def __init__(self, model_path: Union[str, bytes]):

        device_guard()

        if isinstance(model_path, str):
            if not os.path.exists(model_path):
                raise RuntimeError("Model file not found: ", model_path)
            if not os.path.exists(model_path):
                raise RuntimeError()
            model_id, ret = acl.mdl.load_from_file(model_path)
            print(f"load {model_path}")
            check_ret("acl.mdl.load_from_file", ret)
        else:
            assert isinstance(model_path, bytes)
            ptr = acl.util.bytes_to_ptr(model_path)
            model_id, ret = acl.mdl.load_from_mem(ptr, len(model_path))
            check_ret("acl.mdl.load_from_mem", ret)

            model_path = "model in memory"

        self.model_id = model_id

        self.stream = torch.npu.Stream()

        self.npu_stream = self.stream.npu_stream

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, model_id)
        check_ret("acl.mdl.get_desc", ret)

        self.input_sizes = []
        self.output_sizes = []
        self.input_shapes = []
        self.output_shapes = []

        batch, ret = acl.mdl.get_dynamic_batch(self.model_desc)
        check_ret("acl.mdl.get_dynamic_batch", ret)

        self.supported_sorted_batchsize = []
        self.dynamic_index = -1
        if batch['batch']:
            self.batch = {}
            # print(batch)
            self.max_batch_size = batch['batch'][len(batch['batch'])-1]
            for item in batch['batch']:
                self.batch[item] = torch.full(
                    (1,), item,  dtype=torch.int32, device=torch.device('npu:0'))
                self.batch[item] = (
                    self.batch[item], self.batch[item].data_ptr())

            self.dynamic_index, ret = acl.mdl.get_input_index_by_name(
                self.model_desc, 'ascend_mbatch_shape_data')
            check_ret("acl.mdl.get_input_index_by_name", ret)
            assert (self.dynamic_index == 1)

            self.supported_sorted_batchsize = sorted(
                self.batch.keys(), reverse=True)

        for i in range(acl.mdl.get_num_inputs(self.model_desc)):

            dims = acl.mdl.get_input_dims(self.model_desc, i)
            print('\t', dims)
            if dims[0]["name"] == 'ascend_mbatch_shape_data':
                input_size = acl.mdl.get_input_size_by_index(
                    self.model_desc, i)
                print('ascend_mbatch_shape_data: input_size', input_size)
                assert 4 == input_size
                continue
                # ret = acl.mdl.set_dynamic_batch_size(self.model_id, input, 0, 5)

            input_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            self.input_sizes.append(input_size)

            shape = tuple(dims[0]["dims"])
            self.input_shapes.append(shape)
            datatype = acl.mdl.get_input_data_type(self.model_desc, i)
            assert (datatype == 0)

        if self.input_shapes[0][0] != -1:
            assert len(self.supported_sorted_batchsize) == 0
            self.supported_sorted_batchsize = [self.input_shapes[0][0]]
        print('supported_sorted_batchsize', self.supported_sorted_batchsize)

        for i in range(acl.mdl.get_num_outputs(self.model_desc)):
            output_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            self.output_sizes.append(output_size)

            dims = acl.mdl.get_output_dims(self.model_desc, i)
            print('\t', dims)
            shape = list(dims[0]["dims"])
            self.output_shapes.append(shape)
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)
            assert (datatype == 0)

        self.previous_bs = self.input_shapes[0][0]

        print('INPUT: ', self.input_shapes, self.input_sizes)
        print('OUTPUT: ', self.output_shapes, self.output_sizes)

        assert (len(self.input_sizes) == 1)

    def _create_output(self):
        dataset_out = acl.mdl.create_dataset()
        outputs = []

        output_sizes = self.output_sizes
        output_shapes = self.output_shapes

        if self.should_change_bs:
            output_sizes = output_sizes.copy()
            for i in range(len(output_sizes)):
                output_sizes[i] = int(
                    output_sizes[i]/self.max_batch_size * self.should_change_bs)
            output_shapes = output_shapes.copy()
            for i in range(len(output_shapes)):
                output_shapes[i][0] = self.should_change_bs

        for shape, size in zip(output_shapes, output_sizes):
            out = torch.empty(shape, dtype=torch.float32,
                              device=torch.device("npu:0"))
            # torch.npu.synchronize()
            # torch.npu.current_stream().synchronize()
            assert out.is_contiguous()
            # print(out[:1,:1,:1,:1])
            outputs.append(out)
            # print('out: ', out.shape, size)
            data_out = acl.create_data_buffer(out.data_ptr(), size)
            acl.mdl.add_dataset_buffer(dataset_out, data_out)

        return dataset_out, outputs

    def _create_input(self, input_data: torch.Tensor):
        input_data = [input_data]

        dataset_in = acl.mdl.create_dataset()

        input_sizes = self.input_sizes

        if self.should_change_bs:
            #     print(self.model_id, dataset_in, self.dynamic_index, self.should_change_bs)
            #     ret = acl.mdl.set_dynamic_batch_size(self.model_id, dataset_in, self.dynamic_index, self.should_change_bs)
            #     check_ret("acl.mdl.set_dynamic_batch_size", ret)
            self.previous_bs = self.should_change_bs

            input_sizes = input_sizes.copy()
            for i in range(len(input_sizes)):
                input_sizes[i] = int(
                    input_sizes[i]/self.max_batch_size * self.should_change_bs)

        need_sync = False

        for i in range(len(input_data)):
            if (not input_data[i].is_npu) or not input_data[i].dtype == torch.float32:
                input_data[i] = input_data[i].to(
                    torch.device("npu"), dtype=torch.float32)
                need_sync = True
            # if not input_data[i].is_npu:
            #     input_data[i] = input_data[i].npu()
            #     need_sync = True # todo needed for ascend?
            if not input_data[i].is_contiguous():

                input_data[i] = input_data[i].contiguous()

                need_sync = True  # this is a bug for torch_npu

            data_ptr = input_data[i].data_ptr()
            data_in = acl.create_data_buffer(data_ptr, input_sizes[i])
            acl.mdl.add_dataset_buffer(dataset_in, data_in)

        if self.dynamic_index >= 0:
            # assert 4 == self.previous_bs
            data_ptr = self.batch[self.previous_bs][1]
            data_in = acl.create_data_buffer(data_ptr, 4)
            acl.mdl.add_dataset_buffer(dataset_in, data_in)
        if need_sync:
            torch.npu.current_stream().synchronize()

        return dataset_in

    def _split_batch(self, input_data):
        batchsize = input_data.shape[0]

        sp = []
        while batchsize > 0:
            for bs in self.supported_sorted_batchsize:
                if batchsize >= bs:
                    sp.append(bs)
                    batchsize -= bs
                    break
            else:
                raise ValueError(
                    f"Only support batch size in {self.supported_sorted_batchsize}, {batchsize} can not be split")

        re = []
        start = 0
        for bs in sp:
            re.append(input_data[start:start+bs])
            start += bs
        return re

    def forward_list(self, input_data: List[torch.Tensor]):
        assert False
        device_guard()

        assert isinstance(input_data, List)
        in_bs = len(input_data)

        # input_data = [input_data]
        if in_bs in self.supported_sorted_batchsize:
            return self._forward(input_data)

        if in_bs <= self.supported_sorted_batchsize[0]:
            for item in self.supported_sorted_batchsize:
                if in_bs <= item:
                    padding_bs = item
            print(f"padding batch size from {in_bs} to {padding_bs}")
            return self._forward(input_data, padding_bs - in_bs)

        input_datas = self._split_batch(input_data)
        print(
            f"split batch size from {in_bs} to {[x.shape[0] for x in input_datas]}")
        result = []
        for item in input_datas:
            result.append(self._forward(item))
        with torch.npu.stream(self.stream):
            # print("\nconcat ", [x.shape for x in result])
            result = torch.cat(result, dim=0)

            self.stream.synchronize()

        return result

    def __call__(self, input_data: torch.Tensor, padding_batch=True):
        device_guard()

        assert isinstance(input_data, torch.Tensor)

        in_bs = input_data.shape[0]
        # input_data = [input_data]
        if in_bs in self.supported_sorted_batchsize:
            return self._forward(input_data)

        if padding_batch and in_bs <= self.supported_sorted_batchsize[0]:
            for item in self.supported_sorted_batchsize:
                if in_bs <= item:
                    padding_bs = item
            print(f"padding batch size from {in_bs} to {padding_bs}")
            return self._forward(input_data, padding_bs - in_bs)

        input_datas = self._split_batch(input_data)
        # print(f"split batch size from {in_bs} to {[x.shape[0] for x in input_datas]}")
        result = []
        for item in input_datas:
            result.append(self._forward(item))
        with torch.npu.stream(self.stream):
            # print("\nconcat ", [x.shape for x in result])
            result = torch.cat(result, dim=0)

            self.stream.synchronize()

        return result

        # https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC1/developmentguide/appdevg/aclpythondevg/aclpythondevg_0047.html
    def _forward(self, input_data: torch.Tensor, padding_batchsize=0):
        # start  = timer()
        self.should_change_bs = None

        # for inp,inp_sp in zip(input_data, self.input_shapes):
        if True:
            net_shape = self.input_shapes[0]
            input_bs = input_data.shape[0] + padding_batchsize
            if net_shape[0] == -1:
                if self.previous_bs != input_bs:
                    self.should_change_bs = input_bs
                    if input_bs not in self.supported_sorted_batchsize:
                        raise ValueError(
                            f"Only support batch size in {self.supported_sorted_batchsize}")
            elif not tuple(input_data.shape) == net_shape:
                raise ValueError(
                    f"Input shape mismatch: {input_data.shape} vs {net_shape}")

        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            # for tensor created in the stream

            assert (torch.npu.current_stream() == self.stream)

            if padding_batchsize > 0:
                input_data = torch.cat([input_data, torch.empty(
                    (padding_batchsize,) + input_data.shape[1:], dtype=input_data.dtype, device=input_data.device)], dim=0)
            dataset_in = self._create_input(input_data)
            dataset_out, outputs = self._create_output()

            #
            # i donot know why this is needed. bug in torch_npu. todo: check
            self.stream.synchronize()

            # self.dynamic_index = -1
            if self.should_change_bs:
                # acl.rt.set_device(0)
                ret = acl.mdl.set_dynamic_batch_size(
                    self.model_id, dataset_in, 1, self.should_change_bs)
                check_ret("acl.mdl.set_dynamic_batch_size", ret)

            # dims, ret = acl.mdl.get_cur_output_dims(self._desc, index)
            # ret = acl.rt.synchronize_stream(self.npu_stream)
            # self.stream.synchronize()
            # ret = acl.mdl.execute(self.model_id, input, output)
            # ret = acl.mdl.execute_async(self.model_id, data_ptr, output.data_ptr(), stream)
            ret = acl.mdl.execute_async(
                self.model_id, dataset_in, dataset_out, self.npu_stream)

            # ret = acl.mdl.execute(self.model_id, dataset_in, self.dataset_out)

            # ret = acl.rt.synchronize_stream(self.npu_stream)
            self.stream.synchronize()

            destroy_data_set_buffer(dataset_in)
            destroy_data_set_buffer(dataset_out)

            check_ret("acl.rt.synchronize_stream failed", ret)

            if padding_batchsize > 0:
                outputs = [output[:(input_data.shape[0] - padding_batchsize)]
                           for output in outputs]

            if len(outputs) == 1:
                outputs = outputs[0]

            # end = timer() - start
            # print(f"inference {input_data.shape} time: {end}")
            return outputs

    def destroy(self):
        device_guard()
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        print(f"unload {self.model_id}")

        # managed by pytorch
        if False:
            ret = acl.finalize()
            check_ret("acl.finalize", ret)
            print("acl.finalize")


def _get_soc_version():
    """
    获取 soc_version，通过调用系统命令 'npu-smi info -m | grep Ascend'。
    """
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if "Ascend" in line:
                line = line.split("Ascend")[1].strip()
                # print(line)
                return "Ascend"+line
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting soc_version: {e}")
    raise RuntimeError(
        "Failed to get soc_version. result.stdout = ", result.stdout.splitlines())


def onnx2om(onnx_path_or_dir: str, atc_args=''):
    """
    # 动态batchsize(推荐模式)
    python3 acl_helper.py onnx2om /path/to.onnx  --atc_args="--input_shape=input:-1,3,224,224 --dynamic_batch_size=1,2,3,4,8 "
     
     # 静态batchsize
    python src/acl_helper.py onnx2om model/classifier.onnx 
    python src/acl_helper.py onnx2om model/classifier_dynamic.onnx --atc_args="--input_shape=input:1,3,224,224"

    # 强制fp32 （不推荐）
    python src/acl_helper.py onnx2om model/classifier_dynamic.onnx  --atc_args="--precision_mode force_fp32"
    """
    onnxs = []
    if os.path.isfile(onnx_path_or_dir):
        onnxs.append(onnx_path_or_dir)
    else:
        if not os.path.isdir(onnx_path_or_dir):
            raise RuntimeError(
                f"{onnx_path_or_dir} is not an existing file or directory")
        for file in os.listdir(onnx_path_or_dir):
            if file.endswith('.onnx'):
                onnxs.append(os.path.join(onnx_path_or_dir, file))
    assert onnxs, f"No onnx files found in {onnx_path_or_dir}"

    soc = _get_soc_version()

    for file in onnxs:
        print(f'start to process {file}')
        src = file
        dst, file_ext = os.path.splitext(file)

        assert file_ext == '.onnx'

        if "--output" not in atc_args:
            dst += f"_{soc}"
            cmd = f'''atc --framework 5 --soc_version  {soc} --model {src}  --input_format=NCHW --output {dst} --display_model_info 1 --log debug'''
        else:
            cmd = f'''atc --framework 5 --soc_version  {soc} --model {src}  --input_format=NCHW --display_model_info 1 --log debug'''
            dst = atc_args.split("--output")[1].split("--")[0].strip()
            if ".om" in dst:
                dst = dst.replace(".om", "")
                atc_args = atc_args.replace(".om", "")

        if os.path.exists(dst+'.om'):
            raise RuntimeError(f"{dst+'.om'} already exists")

        if atc_args:
            cmd += ' ' + atc_args
        #   --precision_mode        precision mode, support force_fp16(default), force_fp32, cube_fp16in_fp32out, allow_mix_precision, allow_fp32_to_fp16, must_keep_origin_dtype, allow_mix_precision_fp16, allow_mix_precision_bf16, allow_fp32_to_bf16.
        #  --input_shape="input:1,3,244,244"
        else:
            print(
                f"RUNING ATC. Add atc args by passing --atc_args='...'. See atc -h for more options")
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Failed to convert {src} to om")
        else:
            assert os.path.exists(dst+'.om'), f"Failed to convert {src} to om"
            print(f"Saved to {dst+'.om'}")


# https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclpythondevg/aclpythondevg_0059.html
# Usage

def test_infer(model_file: str = ''):
    # model_file = 'model/classifier_dynamic.om'
    if model_file == '':
        model_file = 'model/classifier_dynamic.om'
    model = OmModel(model_file)
    inputs = torch.randn(4, 3, 224, 224)
    outputs = model(inputs)
    print(outputs.shape)
    print(outputs[:, :1])
    for i in range(1, 20):
        inputs = torch.randn(i, 3, 224, 224)
        outputs = model(inputs)
        # print(outputs.shape)
        print(f"i={i} {outputs.shape} \n")


if __name__ == "__main__":
    import fire
    fire.Fire({
        'onnx2om': onnx2om,
        'infer': test_infer
    })
