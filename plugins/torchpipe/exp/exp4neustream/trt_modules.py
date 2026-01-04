
import torch
from typing import Dict, List, Union
import numpy as np
import time
import os
import random
import omniback
# torch.set_grad_enabled(False)

load_params = True
load_params = False
USE_TRT = os.getenv("USE_TRT", "false").lower() in ("true", "1", "yes")
assert USE_TRT
if USE_TRT:
    # import omniback
    # import torchpipe
    print(f"USE_TRT = {USE_TRT}")


class StreamModule(object):
    def __init__(self, device):
        self.device = device
        self.deployed = False
        self.loop_module = False
        self.avg_loop_count = -1

    def set_device(self, device):
        self.device = device

    def set_avg_loop_count(self, avg_loop_count: int):
        self.avg_loop_count = avg_loop_count

    def deploy(self, **kwargs):
        raise NotImplementedError

    def set_implementation(self, **kwargs):
        raise NotImplementedError

    def exec_batch(self, batch_request, **kwargs):
        raise NotImplementedError


class UNetPost:
    def init(self, params, options):
        print(f'UNetPost init, {params}')

    def forward(self, ios):
        data = ios[0]['data']
        self.post_func(data)
        ios[0]['result'] = ios[0]['data']

    def max(self):
        return 1

    def set_function(self, post_func):
        self.post_func = post_func


class UNetModule(StreamModule):
    def __init__(self, device, data_type, parameter_path, scheduler_config: Dict, unet_config: Dict, **kwargs):
        super().__init__(device=device)
        self.scheduler_config = scheduler_config
        self.loop_module = True
        self.avg_loop_count = kwargs["avg_loop_count"]
        self.unet_config = unet_config
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path

        print(f'kwargs={kwargs}')
        # self.instance_index = 0#int(kwargs.pop('instance_index'))
        self.post = UNetPost()
        self.post_pipe = omniback.register(
            f'unet_post', self.post)

        """
        config demo:
        scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'trained_betas': None,
            'prediction_type': 'epsilon',
            'skip_prk_steps': True,
            'set_alpha_to_one': False,
            'steps_offset': 1,
            '_class_name': 'PNDMScheduler',
            '_diffusers_version': '0.6.0',
            'clip_sample': False
        }
        unet_config = {
            "sample_size": 64,
            "down_block_types": (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                ),
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 768,
            "parameter_path": "/path/to/parameter"
        }
        """

    def deploy(self, **kwargs):
        from diffusers import EulerAncestralDiscreteScheduler
        # init scheduler
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.scheduler_config)

        from diffusers.models.unet_2d_condition import UNet2DConditionModel
        
        # init unet
        if USE_TRT:

            import models_omniback
            self.unet_trt = models_omniback.init('unet_trt')

            self.unet = omniback.get(f'unet_trt.0')

            print('unet: max min ', self.unet.max(), self.unet.min())
            self.max_batch_size = self.unet.max()
        else:
            from stable_diffusion_v1_5.stream_module_list.modified_unet_2d_condition import Modified_UNet2DConditionModel
            self.unet = Modified_UNet2DConditionModel(**self.unet_config)
            if load_params:
                self.unet.load_state_dict(torch.load(
                    self.parameter_path, map_location='cpu'))
            # import pdb;pdb.set_trace()
            self.unet = self.unet.to(self.device)
            if self.data_type == torch.float16:
                self.unet.half()

        self.timesteps_and_sigmas = {}
        for i in range(30, 51):
            timestamps, sigma_list = self.get_timesteps_and_sigmas(
                            num_inference_steps=i,
                device='cuda'
                )
            self.timesteps_and_sigmas[i] = (timestamps, sigma_list)
        torch.cuda.synchronize()
        self.deployed = True

    def prepare_latents(self, batch_size, num_inference_steps, height, width, device, dtype, seed=None):
        if True:
            # self.scheduler.set_timesteps(num_inference_steps, device=device)
            self.unet_in_channels = 4
            num_channels_latents = self.unet_in_channels
            shape = (batch_size, num_channels_latents, height // 8, width // 8)
            if seed is not None:
                torch.manual_seed(seed)
            # torch.manual_seed(42)
            latents = torch.randn(
                shape, dtype=dtype, device=device) * self.scheduler.init_noise_sigma
            # print(f'latents.shape = {latents.shape}')
        # print(f'batch_size={batch_size}, {self.latents.shape}')
        return latents

    def scheduler_step(
            self,
            model_output: torch.FloatTensor,
            sigma_list: Union[List[float], torch.FloatTensor],
            sigma_to_list: Union[List[float], torch.FloatTensor],
            sample: torch.FloatTensor) -> torch.FloatTensor:
            # 确保sigma_list和sigma_to_list是张量
            if not isinstance(sigma_list, torch.Tensor):
                    sigma_list = torch.tensor(
                        sigma_list, dtype=sample.dtype, device=sample.device)
            if not isinstance(sigma_to_list, torch.Tensor):
                sigma_to_list = torch.tensor(
                    sigma_to_list, dtype=sample.dtype, device=sample.device)

            # 向量化噪声生成（一次性生成整个batch）
            noise = torch.randn_like(model_output)

            # 根据prediction_type计算pred_original_sample
            if self.scheduler.config.prediction_type == "epsilon":
                pred_original_sample = sample - \
                    sigma_list.view(-1, 1, 1, 1) * model_output
            elif self.scheduler.config.prediction_type == "v_prediction":
                sigma_factor = (-sigma_list / (sigma_list**2 + 1)
                                **0.5).view(-1, 1, 1, 1)
                skip_factor = (1 / (sigma_list**2 + 1)).view(-1, 1, 1, 1)
                pred_original_sample = model_output * sigma_factor + sample * skip_factor

            # 向量化sigma计算
            sigma_from = sigma_list
            sigma_to = sigma_to_list
            sigma_up = (sigma_to**2 * (sigma_from**2 - \
                        sigma_to**2) / sigma_from**2)**0.5
            sigma_down = (sigma_to**2 - sigma_up**2)**0.5

            # 向量化ODE导数计算
            derivative = (sample - pred_original_sample) / \
                          sigma_list.view(-1, 1, 1, 1)
            dt = sigma_down - sigma_list

            # 向量化更新
            prev_sample = sample + derivative * dt.view(-1, 1, 1, 1)
            prev_sample = prev_sample + noise * sigma_up.view(-1, 1, 1, 1)

            return prev_sample

    def scheduler_step_ori(
        self,
        model_output: torch.FloatTensor,
        sigma_list: Union[List[float], torch.FloatTensor],
        sigma_to_list: Union[List[float], torch.FloatTensor],
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        output_list = []
        # torch.cuda.current_stream().synchronize()
        noise_all = torch.randn(
            model_output.shape, dtype=model_output.dtype, device=model_output.device)
        # torch.cuda.current_stream().synchronize()
        for idx in range(model_output.shape[0]):
            if self.scheduler.config.prediction_type == "epsilon":
                pred_original_sample = sample[idx] - \
                    sigma_list[idx] * model_output[idx]
            elif self.scheduler.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output[idx] * (-sigma_list[idx] / (
                    sigma_list[idx]**2 + 1) ** 0.5) + (sample[idx] / (sigma_list[idx]**2 + 1))
            print(
                f'zzz {self.scheduler.config.prediction_type}, {model_output.shape} {sigma_list.shape} {sample.shape}')
            sigma_from = sigma_list[idx]
            sigma_to = sigma_to_list[idx]
            sigma_up = (sigma_to**2 * (sigma_from**2 - \
                        sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

            # 2. Convert to an ODE derivative
            derivative = (sample[idx] - pred_original_sample) / sigma_list[idx]

            dt = sigma_down - sigma_list[idx]

            prev_sample = sample[idx] + derivative * dt

            device = model_output.device
            # torch.cuda.current_stream().synchronize()
            # torch.manual_seed(42)
            # print(f'ssss    {model_output[idx].shape} {model_output.dtype} {device}')
            # noise = torch.randn(model_output[idx].shape, dtype=model_output.dtype, device=device)
            noise = noise_all[idx]

            prev_sample = prev_sample + noise * sigma_up

            output_list.append(prev_sample)
        return torch.stack(output_list)

    def scale_model_input(
        self, sample: torch.FloatTensor, sigma_list: Union[List[float], torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        # sample.shape[0] == timestep.shape[0]
        # 找出来对应index的sigma，然后scale
        sigma_list = sigma_list.reshape(-1, 1, 1, 1)
        sigma_list.square_().add_(1).sqrt_()
        sample = sample / sigma_list
        # for idx in range(sample.shape[0]):
        #     sample[idx] /= ((sigma_list[idx] ** 2 + 1) ** 0.5)
           #print(f"scale factor (divide number) = {((sigma_list[idx] ** 2 + 1) ** 0.5)}")
        # print(f'sample= {sample.shape}, sigma_list={sigma_list.shape}')
        return sample

    def get_timesteps_and_sigmas(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        # added by yhc, to give a request with its relevant sigmas and timesteps
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        timesteps = np.linspace(0, self.scheduler.config.num_train_timesteps - \
                                1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(
            ((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        if self.data_type == torch.float16:
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float16)
            timesteps = torch.from_numpy(timesteps).to(
                device=self.device).to(torch.float16)
        else:
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
            timesteps = torch.from_numpy(timesteps).to(
                device=self.device).to(torch.float32)
        sigmas = torch.from_numpy(sigmas).to(device=self.device)
        return timesteps, sigmas

    def compute(self, batch_request: List[Dict], **kwargs):
        # all_begin = time.perf_counter()
        if not self.deployed:
            raise RuntimeError(
                "UNetModule is not deployed! Can not exec batch!")

        if USE_TRT:
            if len(batch_request) > self.max_batch_size:
                # Split the batch into smaller chunks recursively
                results = []
                chunk_size = self.max_batch_size
                for i in range(0, len(batch_request), chunk_size):
                    chunk = batch_request[i:i + chunk_size]
                    # Recursively call compute on each chunk
                    chunk_results = self.compute(chunk, **kwargs)
                    results.extend(chunk_results)
                return results
        if False:
            for request in batch_request:
                # print(request["loop_index"]["UNetModule"])
                if request["loop_index"]["UNetModule"] == 0:
                    #if "loop_index" not in request:
                    request["latents"] = self.prepare_latents(
                        batch_size=1,
                        num_inference_steps=request["loop_num"]["UNetModule"],
                        height=request["height"],
                        width=request["width"],
                        device=self.device,
                        dtype=self.data_type,
                        seed=request["seed"]
                    )  # .to(self.device)

                    if request["loop_num"]["UNetModule"] not in self.sigma_list:
                        timestamps, sigma_list = self.get_timesteps_and_sigmas(
                            num_inference_steps=request["loop_num"]["UNetModule"],
                            device=self.device
                        )
                        self.sigma_list[request["loop_num"]
                            ["UNetModule"]] = timestamps, sigma_list
                    timestamps, sigma_list = self.sigma_list[request["loop_num"]["UNetModule"]]

                    request["timestamps"] = timestamps
                    request["sigma_list"] = sigma_list
        else:
            # 步骤1: 收集所有需要初始化latents的请求
            requests_to_init = [
                req for req in batch_request if req["loop_index"]["UNetModule"] == 0]

            if requests_to_init:
                # 步骤2: 提取公共参数（假设所有请求的height/width/seed相同）
                first_req = requests_to_init[0]
                batch_size = len(requests_to_init)
                height = first_req["height"]
                width = first_req["width"]
                seed = first_req["seed"]

                # 步骤3: 批量生成latents (一次性生成整个batch)
                latents_batch = self.prepare_latents(
                    batch_size=batch_size,  # 关键修改：使用实际请求数量
                    num_inference_steps=first_req["loop_num"]["UNetModule"],
                    height=height,
                    width=width,
                    device=self.device,
                    dtype=self.data_type,
                    seed=seed
                )

                # 步骤4: 为每个请求分配latents
                for i, req in enumerate(requests_to_init):
                    req["latents"] = latents_batch[i:i+1]  # 保持原batch维度

                    # 单独处理timestamps/sigma（因可能与loop_num相关）
                    # timestamps, sigma_list = self.get_timesteps_and_sigmas(
                    #     num_inference_steps=req["loop_num"]["UNetModule"],
                    #     device=self.device
                    # )
                    timestamps, sigma_list = self.timesteps_and_sigmas[req["loop_num"]["UNetModule"]]
                    req["timestamps"] = timestamps
                    req["sigma_list"] = sigma_list

        # prepare_latents_end = time.perf_counter()
        # print(f"latency for prepare_latent: {prepare_latents_end - all_begin}")

        if True or len(batch_request)*2 not in self.latent_model_input:

            negative_prompt_embeds_list = []
            prompt_embeds_list = []
            latents_list = []
            timestamps_list = []
            guidance_scale_list = []
            sigma_list = []
            sigma_to_list = []

            for request in batch_request:
                negative_prompt_embeds_list.append(request["negative_prompt_embeds"])  # .to(self.device))
                prompt_embeds_list.append(request["prompt_embeds"])  # .to(self.device))
                latents_list.append(request["latents"])  # .to(self.device))
                timestamps_list.append(request["timestamps"][request["loop_index"]["UNetModule"]:request["loop_index"]["UNetModule"] + 1])  # .to(self.device))
                guidance_scale_list.append(request["guidance_scale"])
                sigma_list.append(request["sigma_list"][request["loop_index"]["UNetModule"]:request["loop_index"]["UNetModule"] + 1])  # .to(self.device))
                sigma_to_list.append(request["sigma_list"][request["loop_index"]["UNetModule"] + 1:request["loop_index"]["UNetModule"] + 2])  # .to(self.device))

            # print(f"yhc debug:: timesteps list: {timestamps_list}")
            # 规定multi-dim的tensor，用torch.cat聚合
            # prompt_embeds = torch.cat(
            #     [torch.cat(negative_prompt_embeds_list),
            #     torch.cat(prompt_embeds_list)]
            #     )#.to(self.device)
            prompt_embeds = torch.cat(
                negative_prompt_embeds_list + prompt_embeds_list)

            latents = torch.cat(latents_list)  # .to(self.device)
            timestamps = torch.cat(timestamps_list)  # .to(self.device)
            sigma_list = torch.cat(sigma_list)  # .to(self.device)
            sigma_to_list = torch.cat(sigma_to_list)  # .to(self.device)
            # print(f'timestamps={timestamps.shape}, sigma_list={sigma_list.shape}, sigma_to_list={sigma_to_list.shape}, latents={latents.shape}, prompt_embeds={prompt_embeds.shape}')

            # concat_tensor_end = time.perf_counter()
            # print(f"latency: for concat tensor: {concat_tensor_end - prepare_latents_end}")

            if timestamps.dim() == 0:
                print("error! t should has the size of len(latents)!")
            latent_model_input = self.scale_model_input(latents, sigma_list)

            # print(f'latent_model_input={latent_model_input.shape} {timestamps.shape} {prompt_embeds.shape}')
            # self.latent_model_input[len(batch_request)*2] = (latent_model_input, prompt_embeds, timestamps,guidance_scale_list,sigma_list,sigma_to_list,latents)

            # print(f"latent_model_input= {latent_model_input.shape}")
        # print(f"yhc debug:: scaled latent_model_input: {latent_model_input}")
        # scale_latent_input_end = time.perf_counter()
        # print(f"latency for scale latent model input: {scale_latent_input_end - concat_tensor_end}")

        if USE_TRT:
            bs = timestamps.shape[0]
            # latent_model_input = latent_model_input.reshape(bs,2, 4, 32, 32)
            prompt_embeds = prompt_embeds.reshape(bs, 2,77,768)
            io = omniback.Dict({'data': [latent_model_input, timestamps, prompt_embeds], 'request_size':bs})
            # print(f'unet bs = {bs} ins = {self.instance_index} ')

            self.post.set_function(lambda x : self.unet_postprocess(x, guidance_scale_list, batch_request, sigma_list, sigma_to_list, latents))
            # torch.cuda.current_stream().synchronize()
            self.unet.forward(io, None)

            # print(io['result'].shape)
            # noise_pred = io['result'].reshape(-1, 4, 32, 32)
        else:
            latent_model_input = torch.cat([latent_model_input] * 2)
            if prompt_embeds.shape[0] != latent_model_input.shape[0]:
                print(
                    "Warning! len(prompt_embeds) != len(latents), batch_size is not equal!")
                print(f"yhc debug:: len(prompt_embeds) = {len(prompt_embeds)}")
                print(f"yhc debug:: len(latents) = {len(latent_model_input)}")

            noise_pred = self.unet(
                latent_model_input,
                timestamps,
                encoder_hidden_states=prompt_embeds,
            ).sample
            # print(f"yhc debug:: noise_pred: {noise_pred}")

            # unet_end = time.perf_counter()
            # print(f"latency for unet: {unet_end - scale_latent_input_end}")

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = []
            for idx in range(len(guidance_scale_list)):
                noise_pred.append(noise_pred_uncond[idx] + guidance_scale_list[idx] * (
                    noise_pred_text[idx] - noise_pred_uncond[idx]))
            noise_pred = torch.stack(noise_pred)
            # print(f"yhc debug:: noise_pred after guidance: {noise_pred}")
            # compute the previous noisy sample x_t -> x_t-1

            latents = self.scheduler_step(noise_pred, sigma_list, sigma_to_list, latents)
            # print(f"yhc debug:: latents after scheduler_step: {latents}")

            for idx in range(len(batch_request)):
                batch_request[idx]["latents"] = latents[idx:idx+1]

            # torch.cuda.current_stream().synchronize()

            # step_operation_end = time.perf_counter()
            # print(f"latency for step_operation: {step_operation_end - unet_end}")

            # for idx in range(len(batch_request)):
                # batch_request[idx]["latents"] = latents[idx:idx+1].cpu()
                # batch_request[idx]["loop_index"]["UNetModule"] += 1
                # batch_request[idx]["timestamps"] = batch_request[idx]["timestamps"].cpu()
                # batch_request[idx]["sigma_list"] = batch_request[idx]["sigma_list"].cpu()

            # dispatch_result_end = time.perf_counter()
            # print(f"latency for dispatch result: {dispatch_result_end - step_operation_end}")
            # torch.cuda.current_stream().synchronize()
        return batch_request

    def unet_postprocess(self, noise_pred, guidance_scale_list, batch_request, sigma_list, sigma_to_list, latents):
        noise_pred = noise_pred.reshape(-1, 4, 32, 32)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = []
        for idx in range(len(guidance_scale_list)):
            noise_pred.append(noise_pred_uncond[idx] + guidance_scale_list[idx] * (
                noise_pred_text[idx] - noise_pred_uncond[idx]))
        noise_pred = torch.stack(noise_pred)
        # print(f"yhc debug:: noise_pred after guidance: {noise_pred}")
        # compute the previous noisy sample x_t -> x_t-1

        latents = self.scheduler_step(noise_pred, sigma_list, sigma_to_list, latents)

        for idx in range(len(batch_request)):
            batch_request[idx]["latents"] = latents[idx:idx+1]

        return batch_request


class ClipModule(StreamModule):
    def __init__(self, device: str, data_type, parameter_path, tokenizer_config: Dict, text_encoder_config: Dict, **kwargs):
        super().__init__(device=device)
        self.tokenizer_config = tokenizer_config
        self.text_encoder_config = text_encoder_config
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        """
        tokenizer_config = {
            "vocab_file": "/path/to/file",
            "merges_file": "/path/to/file",
            "model_max_length": int_value,
        }
        text_encoder_config = {
            "clip_text_config_path": "/path/to/file",
            "parameter_path": "/path/to/file",
            "data_type": "data_type" # [torch.float16, torch.float32]
        }
        """

    def deploy(self, **kwargs):
        from transformers.models.clip.tokenization_clip import CLIPTokenizer
        # init tokenizer, according to config
        self.tokenizer = CLIPTokenizer(**self.tokenizer_config
                                       #vocab_file=self.tokenizer_config["vocab_file"],
                                       #merges_file=self.tokenizer_config["merges_file"],
                                       )
        self.tokenizer.model_max_length = self.tokenizer_config["model_max_length"]

        from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextConfig
        # init text_encoder, according to config
        CLIPTextConfig_path = self.text_encoder_config["clip_text_config_path"]
        print(CLIPTextConfig_path)
        self.text_encoder = CLIPTextModel(
            CLIPTextConfig.from_pretrained(CLIPTextConfig_path))
        self.parameter_path = '../model_parameters/hf_format/text_encoder/pytorch_model.bin'

        # self.text_encoder.save_pretrained("a.bin")
        # import pdb;pdb.set_trace()
        if USE_TRT:
            import models_omniback
            self.clip_trt = models_omniback.init('clip_trt')
            instance_index = 0  # *int(kwargs.pop('instance_index'))
            self.text_encoder = (omniback.get(f'clip_trt.{instance_index}'))
            print('clip(text_encoder): max min ',
                  self.text_encoder.max(), self.text_encoder.min())
            self.max_batch_size = self.text_encoder.max()
        else:
            if load_params:
                self.text_encoder.load_state_dict(torch.load(
                    self.parameter_path, map_location="cpu"),  strict=False)
            self.text_encoder = self.text_encoder.to(self.device)
            if self.data_type == torch.float16:
                self.text_encoder = self.text_encoder.half()

        self.deployed = True

    def compute(self, batch_request: List[str], **kwargs):
        if not self.deployed:
            raise RuntimeError(
                "ClipModule is not deployed! Can not exec batch!")

        # print(f'clip bs = {len(batch_request)}')
        if USE_TRT:
            if len(batch_request) > self.max_batch_size:
                # Split the batch into smaller chunks recursively
                results = []
                chunk_size = self.max_batch_size
                for i in range(0, len(batch_request), chunk_size):
                    chunk = batch_request[i:i + chunk_size]
                    # Recursively call compute on each chunk
                    chunk_results = self.compute(chunk, **kwargs)
                    results.extend(chunk_results)
                return results

        batch_prompt = []
        # form the batch data
        for request in batch_request:
            batch_prompt.append(request["prompt"])

        if type(batch_prompt) != list or type(batch_prompt[0]) != str:
            raise RuntimeError("ClipModule.exec should input list of str!")
        batch_size = len(batch_prompt)

        text_inputs = self.tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if False:
            untruncated_ids = self.tokenizer(
                batch_prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
        if USE_TRT:
            attention_mask = None

            # print('text_input_ids', text_input_ids.type(), text_input_ids.device,text_input_ids.shape)
            io = omniback.Dict({'data': text_input_ids, 'node_name': "clip_trt", 'request_size':text_input_ids.shape[0]})
            self.text_encoder.forward(io, None)
            prompt_embeds = io['result']
            # print(f'prompt_embeds {prompt_embeds.dtype} {prompt_embeds.device} {prompt_embeds.shape}')
            # prompt_embeds = prompt_embeds[0]

            # prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
        else:
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            # print(f'text_input_ids={text_input_ids.shape}')
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens: List[str]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if USE_TRT:
            attention_mask = None

            # print('uncond_input.input_ids', uncond_input.input_ids.type(), uncond_input.input_ids.device,uncond_input.input_ids.shape)
            io = omniback.Dict({'data': uncond_input.input_ids, 'node_name': "clip_trt",'request_size':uncond_input.input_ids.shape[0]})
            self.text_encoder.forward(io, None)
            negative_prompt_embeds = io['result']
            # print(f'negative_prompt_embeds {negative_prompt_embeds.dtype} {negative_prompt_embeds.device} {negative_prompt_embeds.shape}')
        else:
            attention_mask = None
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            # import pdb;pdb.set_trace()
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=self.device)

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * 1, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        for idx in range(len(batch_request)):
            batch_request[idx]["negative_prompt_embeds"] = prompt_embeds[idx:idx+1]
            batch_request[idx]["prompt_embeds"] = prompt_embeds[batch_size+ \
                idx:batch_size+idx+1]
        return batch_request


class VaeModule(StreamModule):
    def __init__(self, device, data_type, parameter_path, vae_config: Dict, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        self.vae_config = vae_config
        pass

    def deploy(self, **kwargs):
        # from diffusers.models.autoencoder_kl import AutoencoderKL
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        if USE_TRT:
            import models_omniback
            self.vae_trt = models_omniback.init('vae_trt')
            instance_index = 0  # *int(kwargs.pop('instance_index'))
            self.vae = (omniback.get(f'vae_trt.{instance_index}'))
            print('vae(): max min ', self.vae.max(), self.vae.min())
            self.max_batch_size = self.vae.max()
        else:
            self.vae = AutoencoderKL(**self.vae_config)
            if load_params:
                self.vae.load_state_dict(torch.load(
                    self.parameter_path, map_location='cpu'), strict=False)
            self.vae = self.vae.to(self.device)
            if self.data_type == torch.float16:
                self.vae.half()
            self.max_batch_size = 1600000

        self.deployed = True

    def compute(self, batch_request, **kwargs):
        if not self.deployed:
            raise RuntimeError(
                "VaeModule is not deployed! Can not exec batch!")

        if USE_TRT:
            if len(batch_request) > self.max_batch_size:
                # Split the batch into smaller chunks recursively
                results = []
                chunk_size = self.max_batch_size
                # print(f'use trt chunk_size: {chunk_size}')
                for i in range(0, len(batch_request), chunk_size):
                    chunk = batch_request[i:i + chunk_size]
                    # Recursively call compute on each chunk
                    chunk_results = self.compute(chunk, **kwargs)
                    results.extend(chunk_results)
                return results

        latents = []
        for request in batch_request:
            # print(f'--{request["latents"].shape} {self.max_batch_size}')
            latents.append(request["latents"])
        latents = torch.cat(latents).to(self.device)

        # new vision
        # print(f'latents={latents.shape}')
        if USE_TRT:
            io = omniback.Dict({'data': latents, 'request_size': latents.shape[0]})
            self.vae.forward(io, None)
            images = io['result']
        else:
            images = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]

        for idx in range(len(images)):
            batch_request[idx]["vae_decode_image_tensor"] = images[idx:idx+1]
        return batch_request


class SafetyModule(StreamModule):
    def __init__(self, device, data_type, parameter_path, feature_extractor_config, safety_checker_config, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        self.feature_extractor_config = feature_extractor_config
        self.safety_checker_config = safety_checker_config

        from diffusers.image_processor import VaeImageProcessor
        self.image_processor = VaeImageProcessor()

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(
                    image)
            # print(f'feature_extractor_input={len(feature_extractor_input)} {type(feature_extractor_input[0])}')
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt").to(device)
            # print(
            #     f'safety_checker={image.shape} {safety_checker_input.pixel_values.shape}')
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(
                    dtype)
            )
        return image, has_nsfw_concept

    def deploy(self, **kwargs):
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor
        self.feature_extractor = CLIPImageProcessor(
            **self.feature_extractor_config)
        self.feature_extractor.feature_extractor_type = "CLIPFeatureExtractor"

        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker, CLIPConfig
        if USE_TRT:
            self.safety_checker = None
            # This section contains a mix of CPU and GPU calls that need to be separated
            # to enable overlapping of CPU and GPU computations (pipeline parallelism).
            # Since our current experiment focuses primarily on the iterative UNet process,
            # we'll exclude this optimization for now.
        else:
            self.safety_checker = StableDiffusionSafetyChecker(
                CLIPConfig.from_pretrained(self.safety_checker_config["config_path"]))
            if load_params:
                self.safety_checker.load_state_dict(torch.load(self.safety_checker_config["parameter_path"], map_location='cpu'), strict=False)
            self.safety_checker = self.safety_checker.to(self.device)
            if self.data_type == torch.float16:
                self.safety_checker.half()

        self.deployed = True

    def compute(self, batch_request: List[Dict], **kwargs):
        if not self.deployed:
            raise RuntimeError(
                "ClipModule is not deployed! Can not exec batch!")

        # new version
        for idx in range(len(batch_request)):
            image, has_nsfw_concept = self.run_safety_checker(
                batch_request[idx]["vae_decode_image_tensor"], self.device, self.data_type)
            batch_request[idx]["safety_checked_image_tensor"] = image
            batch_request[idx]["has_nsfw_concept"] = has_nsfw_concept
            if has_nsfw_concept is None:
                do_denormalize = None
            else:
                do_denormalize = [
                    not has_nsfw for has_nsfw in has_nsfw_concept]
            batch_request[idx]["pillow_image"] = self.image_processor.postprocess(
                batch_request[idx]["safety_checked_image_tensor"], output_type="pil", do_denormalize=do_denormalize)[0]
        return batch_request
