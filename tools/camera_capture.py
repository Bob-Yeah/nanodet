# Copyright 2023 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import cv2
import time

def main(camera_id=0, save_dir="./captures"):
    # test_img = cv2.imread("nanodet-plus-arch.png")
    # if test_img is None:
    #     print("错误：无法加载测试图像 'nanodet-plus-arch.png'")
    #     return
    # cv2.imshow("Test Image", test_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开相机
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("错误：无法打开相机")
        return
    
    print(f"相机已成功打开 (ID: {camera_id})")
    print("使用说明：")
    print("  - 按 's' 键拍照")
    print("  - 按 'q' 键退出")
    
    capture_count = 0
    
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法获取图像帧")
            break
        
        # 显示图像
        cv2.imshow('USB Camera', frame)
        
        # 等待按键输入
        key = cv2.waitKey(1) & 0xFF
        
        # 按下 's' 键拍照
        if key == ord('s'):
            # 生成文件名（使用时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            capture_count += 1
            print(f"已拍照 {capture_count}: {filepath}")
        
        # 按下 'q' 键退出
        elif key == ord('q'):
            print("退出程序")
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"共拍摄 {capture_count} 张照片")

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="打开USB相机并实现拍照功能。",
    )
    parser.add_argument(
        "--camera_id", type=int, default=0, help="相机ID (默认打开第一个USB相机)")
    parser.add_argument(
        "--save_dir", type=str, default="./captures", help="照片保存目录")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(camera_id=args.camera_id, save_dir=args.save_dir)