import onnx
from onnx import helper, numpy_helper
import numpy as np

def analyze_onnx_model(model_path):
    """详细分析ONNX模型"""
    model = onnx.load(model_path)
    
    # 数据类型映射
    data_type_map = {
        1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
        6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL", 10: "FLOAT16",
        11: "DOUBLE", 12: "UINT32", 13: "UINT64", 14: "COMPLEX64",
        15: "COMPLEX128"
    }
    
    print("=" * 60)
    print("ONNX模型分析报告")
    print("=" * 60)
    
    # 基本信息
    print(f"\n模型基本信息:")
    print(f"  IR版本: {model.ir_version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print(f"  模型版本: {model.model_version}")
    
    # 统计参数
    total_params = 0
    param_by_type = {}
    param_by_layer = {}
    
    print("\n参数统计:")
    print("-" * 60)
    print(f"{'参数名':<40} {'形状':<20} {'数量':<12} {'数据类型':<10}")
    print("-" * 60)
    
    for initializer in model.graph.initializer:
        shape = tuple(initializer.dims)
        param_count = np.prod(shape) if shape else 1
        data_type = data_type_map.get(initializer.data_type, f"未知({initializer.data_type})")
        
        # 按数据类型统计
        if data_type not in param_by_type:
            param_by_type[data_type] = 0
        param_by_type[data_type] += param_count
        
        # 按层名统计
        param_name = initializer.name
        layer_name = param_name.split('.')[0] if '.' in param_name else param_name
        if layer_name not in param_by_layer:
            param_by_layer[layer_name] = 0
        param_by_layer[layer_name] += param_count
        
        total_params += param_count
        
        # 打印前20个参数
        if len(param_by_layer) <= 20:
            print(f"{param_name:<40} {str(shape):<20} {param_count:<12,} {data_type:<10}")
    
    print("-" * 60)
    print(f"总参数数量: {total_params:,}")
    
    # 计算模型大小
    if 1 in data_type_map:  # FLOAT
        model_size_mb = total_params * 4 / 1024**2
    elif 10 in data_type_map:  # FLOAT16
        model_size_mb = total_params * 2 / 1024**2
    else:
        model_size_mb = total_params * 4 / 1024**2  # 默认float32
    
    print(f"估计模型大小: {model_size_mb:.2f} MB")
    
    # 按层统计
    print("\n按层统计参数:")
    print("-" * 60)
    for layer, count in sorted(param_by_layer.items(), key=lambda x: x[1], reverse=True)[:20]:
        percentage = count / total_params * 100
        print(f"{layer:<30} {count:>12,} ({percentage:>5.1f}%)")
    
    # 按数据类型统计
    print("\n按数据类型统计:")
    print("-" * 60)
    for dtype, count in param_by_type.items():
        percentage = count / total_params * 100
        print(f"{dtype:<10} {count:>12,} ({percentage:>5.1f}%)")
    
    return total_params, param_by_layer

# 使用
model_path = "/home/jiannanye/nanodet/workspace/nanodet_brush_160_0.5x_20251225/nanodet_brush_160_0.5x_20251225_best.onnx"
total_params, layer_stats = analyze_onnx_model(model_path)