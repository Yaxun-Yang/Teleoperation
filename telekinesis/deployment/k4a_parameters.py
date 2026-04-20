from pyk4a import PyK4A, Config
import numpy as np
import pyk4a
import cv2
# 初始化并连接设备 - 使用1536p彩色分辨率和NFOV_UNBINNED深度模式
k4a = PyK4A(Config(
                color_resolution=pyk4a.ColorResolution.RES_1536P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            ))

k4a.start()

# 获取设备校准信息
calibration = k4a.calibration

capture = k4a.get_capture(timeout=33)
            
# Get color image
color = capture.color[..., :3].astype(np.uint8)
print("color image shape: ", color.shape)

# Get depth image
depth = capture.transformed_depth.astype(np.float32) / 1e3
print("origin depth shape: ", depth.shape)

# 获取彩色相机内参矩阵 (参数1表示彩色相机), use this one to convert pc
color_intrinsic_matrix = calibration.get_camera_matrix(1)

# 获取深度相机内参矩阵 (参数0表示深度相机)
depth_intrinsic_matrix = calibration.get_camera_matrix(0)

print("\n彩色相机内参矩阵 (1536p分辨率):")
print(f"fx (x轴焦距) = {color_intrinsic_matrix[0, 0]:.6f}")
print(f"fy (y轴焦距) = {color_intrinsic_matrix[1, 1]:.6f}")
print(f"cx (主点x坐标) = {color_intrinsic_matrix[0, 2]:.6f}")
print(f"cy (主点y坐标) = {color_intrinsic_matrix[1, 2]:.6f}")

print("\n彩色相机内参矩阵 (3x3):")
print(color_intrinsic_matrix)

print("\n深度相机内参矩阵 (NFOV_UNBINNED模式):")
print(f"fx (x轴焦距) = {depth_intrinsic_matrix[0, 0]:.6f}")
print(f"fy (y轴焦距) = {depth_intrinsic_matrix[1, 1]:.6f}")
print(f"cx (主点x坐标) = {depth_intrinsic_matrix[0, 2]:.6f}")
print(f"cy (主点y坐标) = {depth_intrinsic_matrix[1, 2]:.6f}")

print("\n深度相机内参矩阵 (3x3):")
print(depth_intrinsic_matrix)

# 保存彩色相机内参到cam_K.txt（用于点云生成）
cam_K_path = "cam_K.txt"
np.savetxt(cam_K_path, color_intrinsic_matrix, fmt='%.6f')
print(f"\n彩色相机内参矩阵已保存到: {cam_K_path}")

# 保存深度相机内参到depth_cam_K.txt（备用）
depth_cam_K_path = "depth_cam_K.txt"
np.savetxt(depth_cam_K_path, depth_intrinsic_matrix, fmt='%.6f')
print(f"深度相机内参矩阵已保存到: {depth_cam_K_path}")

# 获取分辨率信息
capture = k4a.get_capture()
if capture.color is not None:
    color_height, color_width = capture.color.shape[:2]
    print(f"\n分辨率信息:")
    print(f"彩色相机分辨率: {color_width} x {color_height}")

if capture.depth is not None:
    depth_height, depth_width = capture.depth.shape[:2]
    print(f"深度相机分辨率: {depth_width} x {depth_height}")

# 关闭设备
k4a.stop()