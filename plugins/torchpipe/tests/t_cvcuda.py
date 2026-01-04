# -- coding: utf-8 --
# @Time : 2022/12/28
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
import numpy as np
import time
import torch

# 尝试导入CVCUDA，如果不可用则标记
try:
    import cvcuda
    HAS_CVCUDA = True
except ImportError:
    HAS_CVCUDA = False
    print("CVCUDA not installed. GPU acceleration unavailable.")

# 替代 CVImage 的简单图像处理类


class ImageProcessor:
    @staticmethod
    def read_image(img_path, mode='bgr'):
        """读取图像，默认返回BGR格式"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def show_image(img, window_name='Image', wait_time=0):
        """显示图像"""
        cv2.imshow(window_name, img)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(img, save_path):
        """保存图像"""
        cv2.imwrite(save_path, img)

# 替代 MyTimer 的简单计时器


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"执行时间: {self.interval:.4f} 秒")


def visualize_points(image, points, color=(0, 0, 255), text=True):
    """在图像上可视化点"""
    img_copy = image.copy()
    for i, point in enumerate(points.astype(int)):
        cv2.circle(img_copy, tuple(point), 5, color, -1)
        if text:
            cv2.putText(img_copy, f"P{i+1}:{tuple(point)}", (point[0] + 10, point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy


def get_affine_matrix_from_rotated_rect(src_points, target_w, target_h):
    """
    从旋转矩形的三个点计算仿射矩阵
    :param src_points: 旋转矩形的三个点 [左上, 左下, 右上]
    :param target_w: 目标宽度
    :param target_h: 目标高度
    :return: 仿射变换矩阵
    """
    # 目标点直接由目标宽高决定 (不padding)
    dst_points = np.array([
        [0, 0],           # 左上 -> (0, 0)
        [0, target_h],    # 左下 -> (0, target_h)
        [target_w, 0]     # 右上 -> (target_w, 0)
    ], dtype=np.float32)

    # 计算仿射变换矩阵
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    return affine_matrix, dst_points


def apply_affine_transform_opencv(image, affine_matrix, target_w, target_h, border_value=0):
    """使用OpenCV应用仿射变换"""
    return cv2.warpAffine(
        image,
        affine_matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def apply_affine_transform_cvcuda(image_tensor, affine_matrix, target_w, target_h, border_value=[0, 0, 0]):
    """
    使用CVCUDA应用仿射变换
    :param image_tensor: GPU上的torch tensor，格式为[N, C, H, W]
    :param affine_matrix: 2x3仿射变换矩阵
    :param target_w: 目标宽度
    :param target_h: 目标高度
    :param border_value: 边界填充值 [R, G, B]
    :return: 变换后的图像tensor
    """
    if not HAS_CVCUDA:
        raise RuntimeError("CVCUDA is not available")

    # 转换为NHWC格式 (CVCUDA要求)
    if image_tensor.dim() == 3:  # 如果是单张图像 [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度 [1, C, H, W]
    image_tensor_nhwc = image_tensor.permute(
        0, 2, 3, 1).contiguous()  # [N, H, W, C]

    # 创建cvcuda输入tensor
    cvcuda_input_tensor = cvcuda.as_tensor(image_tensor_nhwc, "NHWC")

    # 创建输出tensor
    device = image_tensor.device
    cvcuda_output_tensor_t = torch.zeros(
        (image_tensor_nhwc.shape[0], target_h,
         target_w, image_tensor_nhwc.shape[3]),
        dtype=torch.uint8,
        device=device
    )
    cvcuda_output_tensor = cvcuda.as_tensor(
        cvcuda_output_tensor_t, layout="NHWC")

    # 执行仿射变换
    cvcuda.warp_affine_into(
        src=cvcuda_input_tensor,
        dst=cvcuda_output_tensor,
        xform=affine_matrix,
        flags=cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.CONSTANT,
        border_value=border_value,
    )

    # 转换回NCHW格式
    result_tensor = cvcuda_output_tensor_t.permute(0, 3, 1, 2).contiguous()

    if result_tensor.shape[0] == 1:  # 如果是单张图像，移除batch维度
        result_tensor = result_tensor.squeeze(0)

    return result_tensor


def compare_results(cpu_result, cuda_result):
    """比较CPU和CUDA版本的结果差异"""
    # 调整尺寸以匹配 (如果需要)
    if cpu_result.shape[:2] != cuda_result.shape[:2]:
        cuda_result = cv2.resize(
            cuda_result, (cpu_result.shape[1], cpu_result.shape[0]))

    if cpu_result.shape == cuda_result.shape:
        diff = np.abs(cpu_result.astype(float) - cuda_result.astype(float))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"CPU vs CUDA 最大差异: {max_diff}")
        print(f"CPU vs CUDA 平均差异: {mean_diff}")

        # 可视化差异
        if diff.ndim == 3:
            diff_gray = np.max(diff, axis=2)
        else:
            diff_gray = diff
        diff_vis = np.clip(diff_gray * 10, 0, 255).astype(np.uint8)
        # 应用颜色映射以便更好地可视化
        diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
        return diff_vis, max_diff, mean_diff
    else:
        print(f"结果尺寸不匹配: CPU {cpu_result.shape} vs CUDA {cuda_result.shape}")
        return None, None, None


def main():
    # 图像路径
    img_p = 'assets/encode_jpeg/grace_hopper_517x606.jpg'  # 替换为实际图像路径

    try:
        # 读取图像
        img = ImageProcessor.read_image(img_p)  # BGR格式
        print(f"图像尺寸: {img.shape}")
    except Exception as e:
        print(f"无法读取图像: {e}")
        # 创建一个测试图像
        print("创建测试图像...")
        img = np.zeros((606, 517, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 500), (255, 128, 64), -1)
        cv2.putText(img, "TEST IMAGE", (150, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    target_w, target_h = 120, 256  # 目标宽高

    # 可视化原图
    ImageProcessor.save_image(img, 'original.jpg')

    # === 基于旋转矩形的三个点计算仿射矩阵 ===
    # 根据图像大小和内容，定义旋转矩形的三个顶点 (左上、左下、右上)
    # 在实际应用中，这些点应该来自检测算法
    h, w = img.shape[:2]

    # 示例1：接近水平的矩形
    src_points = np.array([
        [w * 0.25, h * 0.2],   # 左上角
        [w * 0.2, h * 0.8],    # 左下角
        [w * 0.75, h * 0.25]   # 右上角
    ], dtype=np.float32)

    # 可视化源点
    img_with_src_points = visualize_points(img, src_points, (0, 0, 255))
    ImageProcessor.save_image(img_with_src_points, 'src_points.jpg')

    # 计算仿射矩阵
    mat_, dst_points = get_affine_matrix_from_rotated_rect(
        src_points, target_w, target_h)
    print("计算得到的仿射矩阵:")
    print(mat_)

    # 可视化目标点 (在空白图像上)
    blank_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    blank_with_dst_points = visualize_points(
        blank_img, dst_points, (0, 255, 0))
    ImageProcessor.save_image(blank_with_dst_points, 'dst_points.jpg')

    print("=== OpenCV CPU 版本 ===")
    with Timer() as t:
        for i in range(1000):  # 性能测试
            warped_cpu = apply_affine_transform_opencv(
                img, mat_, target_w, target_h, border_value=0)
            # 只在最后一次迭代保存结果
            if i == 999:
                final_warped_cpu = warped_cpu.copy()

    print(f"OpenCV CPU 版本 - 变换后图像尺寸: {final_warped_cpu.shape}")
    ImageProcessor.save_image(final_warped_cpu, 'cpu_result.jpg')

    # 可视化CPU结果上的目标点
    cpu_result_with_points = visualize_points(
        final_warped_cpu, dst_points, (0, 255, 0))
    ImageProcessor.save_image(cpu_result_with_points,
                              'cpu_result_with_points.jpg')

    # CUDA版本测试
    if HAS_CVCUDA and torch.cuda.is_available():
        print("\n=== CVCUDA GPU 版本 ===")
        # 将图像转换为torch tensor并上传到GPU
        img_tensor = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).cuda()  # [1, C, H, W]
        print(f"输入tensor尺寸: {img_tensor.shape}")

        # 性能测试
        with Timer() as t:
            for i in range(1000):
                warped_gpu_tensor = apply_affine_transform_cvcuda(
                    img_tensor,
                    mat_,
                    target_w,
                    target_h,
                    border_value=[0, 0, 0]  # BGR顺序
                )
                # 只在最后一次迭代保存结果
                if i == 999:
                    final_warped_gpu = warped_gpu_tensor.cpu(
                    ).numpy().transpose(1, 2, 0)  # [H, W, C]

        print(f"CVCUDA GPU 版本 - 变换后图像尺寸: {final_warped_gpu.shape}")
        ImageProcessor.save_image(final_warped_gpu, 'gpu_result.jpg')

        # 比较结果
        diff_vis, max_diff, mean_diff = compare_results(
            final_warped_cpu, final_warped_gpu)
        if diff_vis is not None:
            ImageProcessor.save_image(diff_vis, 'difference.jpg')
            print(f"差异图像已保存为 difference.jpg")
    else:
        print("\n跳过CVCUDA测试，原因:")
        if not HAS_CVCUDA:
            print("- CVCUDA 未安装")
        if not torch.cuda.is_available():
            print("- CUDA 不可用")
        print("要安装CVCUDA，请运行: pip install nvidia-cvcuda")

    print("\n测试完成！结果已保存为:")
    print("- original.jpg: 原始图像")
    print("- src_points.jpg: 原图上的源点")
    print("- dst_points.jpg: 目标点位置")
    print("- cpu_result.jpg: CPU处理结果")
    print("- cpu_result_with_points.jpg: CPU结果上标记目标点")
    if HAS_CVCUDA and torch.cuda.is_available():
        print("- gpu_result.jpg: GPU处理结果")
        print("- difference.jpg: CPU与GPU结果差异")

import cvcuda
class PPLWarpAffineTensor:
    def init(self, params, options):
        print(params, options, 'initit')
        s =  torch.cuda.current_stream()
        
        self.cs = cvcuda.as_stream(s)
        # cvcuda.Stream.current = cs
        print(s, self.cs)
    def forward(self, ios):
        with self.cs:
            print(ios, cvcuda.Stream.current)
            ios[0]['result'] = 3
    def max(self):
        return 1
    

if __name__ == "__main__":
    import omniback as omni
    import torchpipe
    omni.register("affine", PPLWarpAffineTensor())
    a = omni.pipe({"backend": "S[Proxy[affine],SyncTensor]"})
    
    io = {'data':'1'}
    a(io)
    print(io)
    # main()
