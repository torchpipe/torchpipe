import logging
import torch
import numpy as np
import cvcuda

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
    logger = logging.getLogger(__name__)
    logger.info("Running CVCUDA post-processing")
    
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
    if class_probs.device.type != "cuda":
        class_probs = class_probs.cuda()
    
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
def generate_random_batch(batch_size=4, height=517, width=606):
    """
    生成随机输入数据用于测试
    
    返回:
    - probabilities: 随机概率图 [batch, 21, H, W]
    - frame_nhwc: 原始图像 [batch, H, W, 3] (在GPU上)
    - resized_tensor: 预处理图像 [batch, 384, 384, 3] (CVCUDA tensor)
    """
    # 生成随机概率图 (假设21个类别)
    prob_shape = (batch_size, 21, height//2, width//2)
    probabilities = torch.rand(prob_shape, dtype=torch.float32)
    
    # 生成原始图像 (NHWC格式) - 直接在GPU上创建
    frame_nhwc = torch.randint(0, 256, (batch_size, height, width, 3), 
                             dtype=torch.uint8, device="cuda")
    
    # 生成预处理图像 - 在GPU上创建torch张量，然后转换为CVCUDA张量
    resized_tensor_torch = torch.randint(0, 256, (batch_size, 384, 384, 3), 
                                      dtype=torch.uint8, device="cuda")
    resized_tensor = cvcuda.as_tensor(resized_tensor_torch, "NHWC")
    
    return probabilities, frame_nhwc, resized_tensor

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 生成随机输入 (batch=4, 分辨率517x606)
    probs, frames, resized = generate_random_batch(
        batch_size=4,
        height=517,
        width=606
    )
    
    # 执行后处理
    result = postprocess_cvcuda(
        probabilities=probs,
        frame_nhwc=frames,
        resized_tensor=resized,
        class_index=0,  # 选择第一个类别
        output_layout="NHWC",
        gpu_output=True,
        torch_output=True
    )
    
    print("Post-processing completed.")
    print(f"Output shape: {result.shape}, dtype: {result.dtype}, device: {result.device}")
