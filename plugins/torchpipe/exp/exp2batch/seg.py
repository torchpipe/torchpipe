import logging
import torch
import os
from tqdm import tqdm
import numpy as np
import cvcuda
import statistics

def postprocess_cvcuda(
    probabilities: torch.Tensor,
    frame_nhwc: torch.Tensor,
    resized_tensor: cvcuda.Tensor,
    class_index: int,
    output_layout: str = "NHWC",
    gpu_output: bool = True,
    device_id: int = 0,
    torch_output: bool = True,
):
    """
    CVCUDA后处理函数，实现语义分割结果的可视化处理
    
    参数:
    - probabilities: 模型输出的概率图 [batch, classes, H, W]
    - frame_nhwc: 原始输入图像 [batch, H, W, C]
    - resized_tensor: 预处理后的输入图像 (CVCUDA tensor) [batch, resized_H, resized_W, C]
    - class_index: 要可视化的类别索引
    - output_layout: 输出布局 ("NHWC" 或 "NCHW")
    - gpu_output: 是否在GPU上保留输出
    - device_id: GPU设备ID
    - torch_output: 是否返回torch.Tensor
    
    返回: 处理后的图像 (torch.Tensor 或 numpy array)
    """

    # 验证输出布局
    if output_layout not in ["NCHW", "NHWC"]:
        raise ValueError("Invalid output layout: %s" % output_layout)

    # 确保输入在GPU上
    if frame_nhwc.device.type != "cuda":
        frame_nhwc = frame_nhwc.cuda()
    
    # 处理概率图
    actual_batch_size = resized_tensor.shape[0]
    class_probs = probabilities[:actual_batch_size, class_index, :, :]
    class_probs = torch.unsqueeze(class_probs, dim=-1)  # 增加通道维度
    class_probs *= 255
    class_probs = class_probs.type(torch.uint8)
    
    # 确保概率图在GPU上
    assert class_probs.device.type == "cuda"
    
    cvcuda_class_masks = cvcuda.as_tensor(class_probs, "NHWC")

    # 修正：确保批次大小一致
    if frame_nhwc.shape[0] != actual_batch_size:
        frame_nhwc = frame_nhwc[:actual_batch_size]
    
    # 上采样掩码到原始分辨率
    cvcuda_class_masks_upscaled = cvcuda.resize(
        cvcuda_class_masks,
        (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 1),
        cvcuda.Interp.NEAREST,
    )

    # 对低分辨率图像进行高斯模糊
    cvcuda_blurred_input_imgs = cvcuda.gaussian(
        resized_tensor, kernel_size=(15, 15), sigma=(5, 5)
    )
    
    # 上采样模糊图像到原始分辨率
    cvcuda_blurred_input_imgs = cvcuda.resize(
        cvcuda_blurred_input_imgs,
        (frame_nhwc.shape[0],frame_nhwc.shape[1], frame_nhwc.shape[2], 3),
        cvcuda.Interp.LINEAR,
    )

    # 准备引导图像 (灰度图)
    cvcuda_frame_nhwc = cvcuda.as_tensor(frame_nhwc, "NHWC")
    cvcuda_image_tensor_nhwc_gray = cvcuda.cvtcolor(
        cvcuda_frame_nhwc, cvcuda.ColorConversion.RGB2GRAY
    )

    # 应用联合双边滤波平滑边缘
    cvcuda_jb_masks = cvcuda.joint_bilateral_filter(
        cvcuda_class_masks_upscaled,
        cvcuda_image_tensor_nhwc_gray,
        diameter=5,
        sigma_color=50,
        sigma_space=1,
    )

    # 合成最终图像 (前景保留+背景虚化)
    cvcuda_composite_imgs_nhwc = cvcuda.composite(
        cvcuda_frame_nhwc,
        cvcuda_blurred_input_imgs,
        cvcuda_jb_masks,
        3,
    )

    # 转换输出布局
    if output_layout == "NCHW":
        output = cvcuda.reformat(cvcuda_composite_imgs_nhwc, "NCHW")
    else:
        output = cvcuda_composite_imgs_nhwc

    # 转换为所需输出格式
    if gpu_output:
        if torch_output:
            # 直接使用cvcuda张量的底层存储创建torch张量
            output = torch.as_tensor(output.cuda(), device=f"cuda:{device_id}")
    else:
        # 先复制到CPU内存，再转换为numpy
        output = output.cpu().numpy()

    return output

# 生成随机批量化输入数据的函数 - 修复内存问题


def generate_realistic_batch(batch_size=4, height=517, width=606, class_index=0):
    """
    生成更接近真实场景的测试数据
    
    返回:
    - probabilities: 模拟真实分割的概率图 [batch, 21, H, W]
    - frame_nhwc: 原始图像 [batch, H, W, 3]
    - resized_tensor: 预处理图像 [batch, 384, 384, 3] (CVCUDA tensor)
    """
    # 概率图分辨率是原始图像的一半
    h_prob, w_prob = height // 2, width // 2

    # 创建固定模式的掩码（中心圆形区域）
    center_y, center_x = h_prob // 2, w_prob // 2
    radius = min(h_prob, w_prob) // 4

    # 生成网格坐标
    y, x = torch.meshgrid(
        torch.arange(h_prob, device="cuda"),
        torch.arange(w_prob, device="cuda"),
        indexing='ij'
    )

    # 计算距离中心点的欧氏距离
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)

    # 创建圆形掩码 (0.0-1.0)
    circle_mask = torch.where(dist <= radius, 0.95, 0.05).float()

    # 添加随机噪声模拟真实预测
    noise = torch.rand((batch_size, 1, h_prob, w_prob), device="cuda") * 0.1
    class0_probs = (circle_mask + noise).clamp(0, 1)

    # 创建完整概率图 (batch, 21, H, W)
    probabilities = torch.zeros((batch_size, 21, h_prob, w_prob),
                                device="cuda", dtype=torch.float32)

    # 将类别0的概率设置为圆形区域
    probabilities[:, class_index] = class0_probs.squeeze(1)

    # 其他类别使用剩余概率 (更真实的分布)
    remaining_probs = (1 - class0_probs) / 20
    for i in range(21):
        if i != class_index:
            probabilities[:, i] = remaining_probs.squeeze(1)

    # 生成原始图像 (NHWC格式)
    frame_nhwc = torch.randint(0, 256, (batch_size, height, width, 3),
                               dtype=torch.uint8, device="cuda")

    # 生成预处理图像 (CVCUDA tensor)
    resized_tensor_torch = torch.randint(0, 256, (batch_size, 384, 384, 3),
                                         dtype=torch.uint8, device="cuda")
    resized_tensor = cvcuda.as_tensor(resized_tensor_torch, "NHWC")

    return probabilities, frame_nhwc, resized_tensor


def main(batch_size=1, gpu_id=0, total=4000, img_path='../../tests/assets/encode_jpeg/'):
    # 配置日志    
    
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        assert str(gpu_id) == os.environ['CUDA_VISIBLE_DEVICES'] or gpu_id == 0
        gpu_id = 0
    elif gpu_id != 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        gpu_id = 0
    torch.cuda.set_device(gpu_id)
    # 生成随机输入 (batch=4, 分辨率517x606)
    probs, frames, resized = generate_realistic_batch(
        batch_size=batch_size,
        height=517,
        width=606,
        class_index=0
    )
    
    for _ in range(5):
        result = postprocess_cvcuda(
            probabilities=probs,
            frame_nhwc=frames,
            resized_tensor=resized,
            class_index=0,  # 选择第一个类别
            output_layout="NHWC",
            gpu_output=True,
            torch_output=True,
            device_id=gpu_id,
        )
    torch.cuda.synchronize()
    print('Warm-up finished')
    print(
        f"Output shape: {result.shape}, dtype: {result.dtype}, device: {result.device}")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iteration_times = []

    for _ in tqdm(range(total)):
        start_event.record()
        result = postprocess_cvcuda(
            probabilities=probs,
            frame_nhwc=frames,
            resized_tensor=resized,
            class_index=0,  # 选择第一个类别
            output_layout="NHWC",
            gpu_output=True,
            torch_output=True,
            device_id=gpu_id,
        )
        end_event.record()
        torch.cuda.current_stream().synchronize()  # Ensure measurement completes
        elapsed_ms = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_ms / 1000.0)  # Convert to seconds

    # Calculate statistics
    median_time_per_batch = statistics.median(iteration_times)
    images_per_second = batch_size / median_time_per_batch
    batches_per_second = 1 / median_time_per_batch

    print(f"\nBenchmark Results (GPU {gpu_id}):")
    print(f"- Total batches processed: {total}")
    print(f"- Total images decoded: {batch_size * total}")
    print(f"- Median time per batch: {median_time_per_batch * 1000:.4f} ms")
    print(f"- Throughput: {batches_per_second:.2f} qps")



# 使用示例
if __name__ == "__main__":

    import fire
    fire.Fire(main)