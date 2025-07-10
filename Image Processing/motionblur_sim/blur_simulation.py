import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def create_motion_blur_kernel(size, angle):
    """
    创建一个用于运动模糊的卷积核。
    
    参数:
    size (int): 核心的大小（奇数），代表模糊的长度。
    angle (float): 运动的角度（0-360度）。
    
    返回:
    numpy.ndarray: 归一化后的运动模糊卷积核。
    """
    kernel = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)

    for i in range(size):
        x = int(round(center + (i - center) * dx))
        y = int(round(center + (i - center) * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
            
    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        kernel[center, center] = 1
    else:
        kernel = kernel / kernel_sum
        
    return kernel

def apply_blur_to_image(args):
    """
    对单个图片应用运动模糊效果并保存。
    这是一个工作函数，为并行处理设计。
    
    参数:
    args (tuple): 包含 (input_path, output_path, kernel) 的元组。
    """
    input_path, output_path, kernel = args
    try:
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            # print(f"警告：无法读取文件 {os.path.basename(input_path)}，跳过。")
            return f"读取失败: {os.path.basename(input_path)}"
        
        # 应用卷积进行模糊处理，并使用 BORDER_REPLICATE 模式避免黑边
        blurred_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        
        # 保存处理后的图片
        cv2.imwrite(output_path, blurred_image)
        return None # 表示成功
    except Exception as e:
        # print(f"错误：处理文件 {os.path.basename(input_path)} 时发生错误: {e}")
        return f"处理错误: {os.path.basename(input_path)}"

def batch_process_images(input_dir, output_dir, blur_strength, blur_angle):
    """
    批量处理目录中的所有图片，应用运动模糊，并使用并行处理加速。

    参数:
    input_dir (str): 包含原始图片的目录路径。
    output_dir (str): 用于保存处理后图片的目录路径。
    blur_strength (int): 模糊的强度（模糊核的大小，必须是奇数）。
    blur_angle (float): 模糊的角度。
    """
    # 验证模糊强度必须为正奇数
    if blur_strength <= 0 or blur_strength % 2 == 0:
        print("错误：模糊强度 (blur_strength) 必须是一个正奇数。例如：15, 31, 51。")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义支持的图片文件扩展名
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 查找所有支持的图片文件
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    except FileNotFoundError:
        print(f"错误：输入目录 '{input_dir}' 不存在。")
        return

    if not image_files:
        print(f"在目录 '{input_dir}' 中没有找到支持的图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片。开始处理...")

    # 预先创建卷积核，所有进程共享同一个核
    motion_kernel = create_motion_blur_kernel(blur_strength, blur_angle)

    # 准备任务列表，每个任务是一个包含输入、输出路径和核的元组
    tasks = [(os.path.join(input_dir, filename), os.path.join(output_dir, filename), motion_kernel) for filename in image_files]

    # 使用 ProcessPoolExecutor 进行并行处理
    # max_workers=None 会自动设置为本机的CPU核心数
    with ProcessPoolExecutor(max_workers=None) as executor:
        # 使用tqdm创建进度条
        results = list(tqdm(executor.map(apply_blur_to_image, tasks), total=len(tasks), desc="应用运动模糊"))

    # 统计处理结果
    success_count = results.count(None)
    failure_count = len(results) - success_count
    print("\n处理完成！")
    print(f"成功处理图片: {success_count} 张")
    if failure_count > 0:
        print(f"失败或跳过: {failure_count} 张")


# ==============================================================================
# --- 程序主入口：请在这里配置您的参数 ---
# ==============================================================================
if __name__ == "__main__":
    # 1. 设置输入和输出目录
    # 请使用您的实际路径，Windows路径示例: "C:\\Users\\YourUser\\Desktop\\Input"
    INPUT_DIRECTORY = "/Users/daijun/Downloads/sharp_imgs"
    OUTPUT_DIRECTORY = "/Users/daijun/Downloads/blurred_imgs_41"

    # 2. 控制运动模糊效果的参数
    # BLUR_STRENGTH: 模糊强度/长度。必须是正奇数。数值越大，模糊越明显。
    BLUR_STRENGTH = 41
    
    # BLUR_ANGLE: 模糊角度 (0-360)。0是水平，45是斜向，90是垂直。
    BLUR_ANGLE = 30.0

    # 运行批量处理函数
    batch_process_images(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        blur_strength=BLUR_STRENGTH,
        blur_angle=BLUR_ANGLE
    )