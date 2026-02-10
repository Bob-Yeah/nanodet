import json
import numpy as np
data = {
    "behind_camera": False,
    "truncation": 0.0,
    "bbox2D_tight": [-1, -1, -1, -1],
    "visibility": -1,
    "segmentation_pts": -1,
    "lidar_pts": -1,
    "valid3D": True,
    "category_id": 28,
    "category_name": "sink",
    "id": 2828167,
    "image_id": 173384,
    "dataset_id": 16,
    "bbox2D_proj": [317.7158508300781, 545.8804931640625, 828.8178100585938, 798.7599487304688],
    "depth_error": 0.6789064095224648,
    "bbox2D_trunc": [317.7158508300781, 545.8804931640625, 828.8178100585938, 798.7599487304688],
    "center_cam": [-0.15078339540062935, -0.3835346408544088, 1.9770693868850286],
    "dimensions": [0.5454616843414527, 0.18081009258798972, 0.39088701292880146],
    "bbox3D_cam": [[-0.44767552614212036, -0.4166813790798187, 1.7995597124099731], [-0.10029754042625427, -0.344592809677124, 1.6354730129241943], [-0.10376906394958496, -0.17649266123771667, 1.7019755840301514], [-0.45114704966545105, -0.24858124554157257, 1.8660621643066406], [-0.1977977156639099, -0.5905766487121582, 2.2521631717681885], [0.14958027005195618, -0.5184880495071411, 2.088076591491699], [0.1461087465286255, -0.35038790106773376, 2.1545791625976562], [-0.2012692391872406, -0.42247647047042847, 2.3186657428741455]],
    "R_cam": [[0.8886915554030996, -0.019199757939725132, 0.4581033602259165], [0.18442304821140873, 0.9297054833226549, -0.3188037853730743], [-0.419780250421137, 0.3678030499803455, 0.8297622899249658]]
}

# {"behind_camera": false, "truncation": 0.0, "bbox2D_tight": [-1, -1, -1, -1], "visibility": -1, "segmentation_pts": -1, "lidar_pts": -1, "valid3D": true, "category_id": 28, "category_name": "sink", "id": 2828167, "image_id": 173384, "dataset_id": 16, "bbox2D_proj": [317.7158508300781, 545.8804931640625, 828.8178100585938, 798.7599487304688], "depth_error": 0.6789064095224648, "bbox2D_trunc": [317.7158508300781, 545.8804931640625, 828.8178100585938, 798.7599487304688], "center_cam": [-0.15078339540062935, -0.3835346408544088, 1.9770693868850286], "dimensions": [0.5454616843414527, 0.18081009258798972, 0.39088701292880146], "bbox3D_cam": [[-0.44767552614212036, -0.4166813790798187, 1.7995597124099731], [-0.10029754042625427, -0.344592809677124, 1.6354730129241943], [-0.10376906394958496, -0.17649266123771667, 1.7019755840301514], [-0.45114704966545105, -0.24858124554157257, 1.8660621643066406], [-0.1977977156639099, -0.5905766487121582, 2.2521631717681885], [0.14958027005195618, -0.5184880495071411, 2.088076591491699], [0.1461087465286255, -0.35038790106773376, 2.1545791625976562], [-0.2012692391872406, -0.42247647047042847, 2.3186657428741455]], "R_cam": [[0.8886915554030996, -0.019199757939725132, 0.4581033602259165], [0.18442304821140873, 0.9297054833226549, -0.3188037853730743], [-0.419780250421137, 0.3678030499803455, 0.8297622899249658]]}

# 解析JSON数据
# data = {
#     "behind_camera": True,
#     "truncation": -1,
#     "bbox2D_tight": [-1, -1, -1, -1],
#     "visibility": -1,
#     "segmentation_pts": -1,
#     "lidar_pts": -1,
#     "valid3D": True,
#     "category_id": 30,
#     "category_name": "bathtub",
#     "id": 2828168,
#     "image_id": 173384,
#     "dataset_id": 16,
#     "bbox2D_proj": [1439.0, 1071.874755859375, 10582.9208984375, 9019.4619140625],
#     "depth_error": -1,
#     "center_cam": [2.0457224852667597, 0.6565420189507538, 0.6332483462697214],
#     "dimensions": [1.655427647516021, 0.5089507128903599, 0.7142064990653796],
#     "bbox3D_cam": [[2.7683517932891846, 0.23307910561561584, 1.052625060081482], [2.146409273147583, 0.09279817342758179, 1.3744943141937256], [2.136245012283325, 0.5661708116531372, 1.5611649751663208], [2.7581875324249268, 0.7064516544342041, 1.2392957210540771], [1.955199956893921, 0.7469131946563721, -0.2946683168411255], [1.3332574367523193, 0.6066323518753052, 0.02720099687576294], [1.3230931758880615, 1.0800049304962158, 0.21387162804603577], [1.945035696029663, 1.2202858924865723, -0.10799771547317505]],
#     "R_cam": [[-0.8708158744334646, -0.019971025401610026, -0.49120349243372563], [-0.1964150734293135, 0.9300951265479618, 0.31039357999399536], [0.45066709638539004, 0.36677542681610964, -0.813864211351883]],
#     "bbox2D_trunc": [-1, -1, -1, -1]
# }

# 提取关键数据
center_cam = np.array(data['center_cam'])
dimensions = np.array(data['dimensions'])
bbox3D_cam = np.array(data['bbox3D_cam'])
R_cam = np.array(data['R_cam'])

print("=== 验证1: bbox3D_cam的中心是否与center_cam一致 ===")
# 计算bbox3D_cam的中心点
bbox3D_center = np.mean(bbox3D_cam, axis=0)
print(f"bbox3D_cam的中心点: {bbox3D_center}")
print(f"center_cam: {center_cam}")
print(f"差值: {np.abs(bbox3D_center - center_cam)}")
print(f"是否一致: {np.allclose(bbox3D_center, center_cam, atol=1e-3)}")

print("\n=== 验证2: 计算理论bbox3D_cam并与实际比较 ===")
# 生成立方体的8个顶点（在物体坐标系中）
half_dims = dimensions / 2
object_corners = np.array([
    [ half_dims[0], -half_dims[1],  half_dims[2]],  # 前上右
    [ half_dims[0], -half_dims[1], -half_dims[2]],  # 后上右
    [-half_dims[0], -half_dims[1], -half_dims[2]],  # 后上左
    [-half_dims[0], -half_dims[1],  half_dims[2]],  # 前上左
    [ half_dims[0],  half_dims[1],  half_dims[2]],  # 前下右
    [ half_dims[0],  half_dims[1], -half_dims[2]],  # 后下右
    [-half_dims[0],  half_dims[1], -half_dims[2]],  # 后下左
    [-half_dims[0],  half_dims[1],  half_dims[2]],  # 前下左
])

# 转换到相机坐标系（旋转 + 平移）
camera_corners = np.dot(object_corners, R_cam.T) + center_cam
print("\n理论计算的bbox3D_cam:")
for i, corner in enumerate(camera_corners):
    print(f"顶点{i}: {corner}")

print("\n实际的bbox3D_cam:")
for i, corner in enumerate(bbox3D_cam):
    print(f"顶点{i}: {corner}")

print("\n比较结果:")
for i in range(8):
    diff = np.abs(camera_corners[i] - bbox3D_cam[i])
    print(f"顶点{i}差值: {diff}, 是否一致: {np.allclose(camera_corners[i], bbox3D_cam[i], atol=1e-3)}")

print(f"\n整体是否一致: {np.allclose(camera_corners, bbox3D_cam, atol=1e-3)}")

print("\n=== 验证3: bbox3D_cam和dimensions是否一致 ===")
# 计算bbox3D_cam的尺寸范围
min_corner = np.min(bbox3D_cam, axis=0)
max_corner = np.max(bbox3D_cam, axis=0)
calculated_dims = max_corner - min_corner
print(f"根据bbox3D_cam计算的尺寸: {calculated_dims}")
print(f"原始dimensions: {dimensions}")
print(f"差值: {np.abs(calculated_dims - dimensions)}")
print(f"是否一致: {np.allclose(calculated_dims, dimensions, atol=1e-3)}")
