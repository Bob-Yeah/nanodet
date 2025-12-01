#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
from collections import defaultdict


def split_coco_json(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将COCO格式的JSON文件按比例分割成train.json, val.json和test.json
    确保category_id为1和2的annotations也大致按相同比例分布
    """
    # 确保比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例和必须为1"
    
    # 创建输出目录
    output_dir = os.path.dirname(input_file)
    
    # 读取输入文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    # 按category_id分组annotations
    cat1_annotations = [ann for ann in annotations if ann['category_id'] == 0]
    cat2_annotations = [ann for ann in annotations if ann['category_id'] == 1]
    other_annotations = [ann for ann in annotations if ann['category_id'] not in [0, 1]]
    
    # 获取每个类别的image_ids
    cat1_image_ids = set(ann['image_id'] for ann in cat1_annotations)
    cat2_image_ids = set(ann['image_id'] for ann in cat2_annotations)

    print(cat1_image_ids)
    print("-"*30)
    print(cat2_image_ids)
    
    print(f"总图像数: {len(images)}")
    print(f"总标注数: {len(annotations)}")
    print(f"类别1标注数: {len(cat1_annotations)}, 对应图像数: {len(cat1_image_ids)}")
    print(f"类别2标注数: {len(cat2_annotations)}, 对应图像数: {len(cat2_image_ids)}")
    print(f"其他类别标注数: {len(other_annotations)}")
    
    # 将图像分为三类：包含cat1的、包含cat2的、其他
    # cat1_images = [img for img in images if img['id'] in cat1_image_ids]
    
    cat2_images = []
    for _img in images:
        if _img['id'] in cat2_image_ids or _img['id'] in cat1_image_ids:
            if (_img not in cat2_images):
                cat2_images.append(_img)

    other_images = []
    for _img in images:
        if _img not in cat2_images:
            if (_img not in other_images):
                other_images.append(_img)

    print(f"类别2图像数: {len(cat2_images)}")
    print(f"其他图像数: {len(other_images)}")
    # return 
    # 分别对每类图像进行随机打乱和分割
    random.seed(42)  # 设置随机种子以确保结果可复现
    
    def split_list(items, train_r, val_r, test_r):
        """按比例分割列表"""
        random.shuffle(items)
        total = len(items)
        train_count = int(total * train_r)
        val_count = int(total * val_r)
        
        train = items[:train_count]
        val = items[train_count:train_count + val_count]
        test = items[train_count + val_count:]
        
        return train, val, test
    
    # 分别分割三类图像
    # cat1_train, cat1_val, cat1_test = split_list(cat1_images, train_ratio, val_ratio, test_ratio)
    cat2_train, cat2_val, cat2_test = split_list(cat2_images, train_ratio, val_ratio, test_ratio)
    other_train, other_val, other_test = split_list(other_images, train_ratio, val_ratio, test_ratio)
    
    # 合并分割后的图像
    train_images = cat2_train + other_train
    val_images = cat2_val + other_val
    test_images = cat2_test + other_test
    
    # 获取各集合的image_ids
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)
    test_image_ids = set(img['id'] for img in test_images)
    
    # 根据image_ids分配annotations
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
    
    # 统计各集合中各类别的annotations数量
    def count_by_category(anns):
        counts = defaultdict(int)
        for ann in anns:
            counts[ann['category_id']] += 1
        return counts
    
    train_counts = count_by_category(train_annotations)
    val_counts = count_by_category(val_annotations)
    test_counts = count_by_category(test_annotations)
    
    print("\n分割结果:")
    print(f"Train: {len(train_images)}张图像, {len(train_annotations)}个标注")
    print(f"  类别分布: {dict(train_counts)}")
    print(f"Val: {len(val_images)}张图像, {len(val_annotations)}个标注")
    print(f"  类别分布: {dict(val_counts)}")
    print(f"Test: {len(test_images)}张图像, {len(test_annotations)}个标注")
    print(f"  类别分布: {dict(test_counts)}")
    
    # 创建输出数据结构
    def create_output_data(images_subset, annotations_subset):
        output_data = {
            'images': images_subset,
            'annotations': annotations_subset,
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data.get('categories', [])
        }
        return output_data
    
    # 创建三个分割文件
    train_data = create_output_data(train_images, train_annotations)
    val_data = create_output_data(val_images, val_annotations)
    test_data = create_output_data(test_images, test_annotations)
    
    # 保存文件 - 添加异常处理和更明确的路径
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    # 保存函数，添加错误处理
    def save_json(data, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"成功保存: {file_path}")
            return True
        except PermissionError:
            print(f"错误: 没有写入权限到 {file_path}")
            # 尝试保存到用户主目录
            fallback_path = os.path.join(os.path.expanduser('~'), os.path.basename(file_path))
            print(f"尝试保存到: {fallback_path}")
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"成功保存到备用路径: {fallback_path}")
                return True
            except Exception as e:
                print(f"保存到备用路径失败: {e}")
                return False
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    # 保存所有文件
    save_json(train_data, train_file)
    save_json(val_data, val_file)
    save_json(test_data, test_file)
    
    print(f"\n文件已保存:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    
    return train_file, val_file, test_file


if __name__ == '__main__':
    # 设置输入文件路径
    input_json = os.path.join(os.path.dirname(__file__), 'result.json')
    
    # 执行分割
    split_coco_json(input_json)