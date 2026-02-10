import os
import cv2
import numpy as np
import argparse
import glob

class ImagePreprocessor:
    def __init__(self, target_size, flip_type='horizontal', pad_value=0):
        """
        初始化图像预处理器
        
        Args:
            target_size: 目标尺寸，格式为 (width, height)
            flip_type: 翻转类型，可选值: 'horizontal' (水平翻转), 'vertical' (垂直翻转), 'both' (水平和垂直翻转)
            pad_value: 填充值，默认为0（黑色）
        """
        self.target_size = target_size
        self.flip_type = flip_type
        self.pad_value = pad_value
    
    def flip_image(self, image):
        """
        翻转图像
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            翻转后的图像
        """
        if self.flip_type == 'horizontal':
            return cv2.flip(image, 1)  # 水平翻转
        elif self.flip_type == 'vertical':
            return cv2.flip(image, 0)  # 垂直翻转
        elif self.flip_type == 'both':
            return cv2.flip(image, -1)  # 水平和垂直翻转
        else:
            return image  # 不翻转
    
    def pad_and_center(self, image):
        """
        将图像调整到目标尺寸：对于每个维度，如果图像尺寸大于目标尺寸则裁剪，否则填充
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            调整后的图像
        """
        target_width, target_height = self.target_size
        h, w = image.shape[:2]
        
        # 复制原始图像
        result = image.copy()
        
        # 处理宽度方向
        if w < target_width:
            # 填充宽度
            pad_left = (target_width - w) // 2
            pad_right = target_width - w - pad_left
            result = cv2.copyMakeBorder(result, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[self.pad_value]*3)
        
        # 更新宽度
        w = target_width
        
        # 处理高度方向
        if h < target_height:
            # 填充高度
            pad_top = (target_height - h) // 2
            pad_bottom = target_height - h - pad_top
            result = cv2.copyMakeBorder(result, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[self.pad_value]*3)
        
        return result
    
    def crop_and_center(self, image):
        """
        裁剪图像到目标尺寸，保持比例并居中裁剪
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            裁剪后的图像
        """
        target_width, target_height = self.target_size
        h, w = image.shape[:2]
        
        # 计算裁剪比例
        scale = max(target_width / w, target_height / h)
        new_w = int(target_width / scale)
        new_h = int(target_height / scale)
        
        # 计算裁剪位置（居中裁剪）
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        
        # 裁剪图像
        cropped_image = image[start_y:start_y+new_h, start_x:start_x+new_w]
        
        # 调整到目标尺寸
        result = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        return result
    
    def auto_adjust(self, image):
        """
        自动调整图像尺寸：对于每个维度，如果图像尺寸大于目标尺寸则裁剪，否则填充
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            调整后的图像
        """
        target_width, target_height = self.target_size
        h, w = image.shape[:2]
        
        # 复制原始图像
        result = image.copy()
        
        # 处理宽度方向
        if w > target_width:
            # 裁剪宽度
            start_x = (w - target_width) // 2
            result = result[:, start_x:start_x+target_width]
        elif w < target_width:
            # 填充宽度
            pad_left = (target_width - w) // 2
            pad_right = target_width - w - pad_left
            result = cv2.copyMakeBorder(result, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[self.pad_value]*3)
        
        # 更新宽度
        w = target_width
        
        # 处理高度方向
        if h > target_height:
            # 裁剪高度
            start_y = (h - target_height) // 2
            result = result[start_y:start_y+target_height, :]
        elif h < target_height:
            # 填充高度
            pad_top = (target_height - h) // 2
            pad_bottom = target_height - h - pad_top
            result = cv2.copyMakeBorder(result, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[self.pad_value]*3)
        
        return result
    
    def direct_resize(self, image):
        """
        直接调整图像尺寸到目标大小，不保持比例
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            调整后的图像
        """
        target_width, target_height = self.target_size
        # 直接调整图像尺寸
        result = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return result
    
    def process_image(self, image_path, output_path, mode='pad'):
        """
        处理单张图像：读取 -> 翻转 -> 调整尺寸
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            mode: 调整尺寸模式，可选值: 'pad' (填充/裁剪), 'crop' (保持比例裁剪), 'auto' (自动), 'resize' (直接调整尺寸)
            
        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot read image {image_path}")
                return False
            
            # 翻转图像
            flipped_image = self.flip_image(image)
            
            # 调整尺寸
            if mode == 'pad':
                processed_image = self.pad_and_center(flipped_image)
            elif mode == 'crop':
                processed_image = self.crop_and_center(flipped_image)
            elif mode == 'auto':
                processed_image = self.auto_adjust(flipped_image)
            elif mode == 'resize':
                processed_image = self.direct_resize(flipped_image)
            else:
                print(f"Error: Invalid mode {mode}")
                return False
            
            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed_image)
            
            return True
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def process_folder(self, input_folder, output_folder, mode='pad', extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """
        处理文件夹中的所有图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            mode: 调整尺寸模式，可选值: 'pad' (填充), 'crop' (裁剪)
            extensions: 要处理的图像扩展名列表
            
        Returns:
            tuple: (成功处理的图像数量, 总图像数量)
        """
        # 获取所有图像文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
        
        total = len(image_paths)
        success = 0
        
        print(f"Found {total} images in {input_folder}")
        print(f"Processing images...")
        
        for i, image_path in enumerate(image_paths):
            # 生成输出路径
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, filename)
            
            # 处理图像
            if self.process_image(image_path, output_path, mode):
                success += 1
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Progress: {i + 1}/{total} images processed")
        
        print(f"Processing completed: {success}/{total} images successfully processed")
        return success, total

def parse_args():
    parser = argparse.ArgumentParser(description='Image Preprocessing Tool')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output folder for processed images')
    parser.add_argument('--target-size', '-s', type=int, nargs=2, required=True, help='Target size (width height)')
    parser.add_argument('--mode', '-m', type=str, choices=['pad', 'crop', 'auto', 'resize'], default='auto', help='Resize mode: pad (fill/crop), crop (maintain aspect ratio), auto (auto adjust), resize (direct resize)')
    parser.add_argument('--flip', '-f', type=str, choices=['horizontal', 'vertical', 'both', 'none'], default='horizontal', help='Flip type')
    parser.add_argument('--pad-value', '-p', type=int, default=0, help='Padding value (0-255)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(
        target_size=args.target_size,
        flip_type=args.flip,
        pad_value=args.pad_value
    )
    
    # 处理文件夹
    success, total = preprocessor.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        mode=args.mode
    )
    
    if success == total:
        print("All images processed successfully!")
    else:
        print(f"Some images failed to process. Success rate: {success/total*100:.2f}%")

if __name__ == '__main__':
    main()