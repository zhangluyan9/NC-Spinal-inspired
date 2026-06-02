import numpy as np

# 加载数据
data = np.load('robot_data_projected.npz')
X = data['X']
y = data['y']

# 确保 y 是一维的（避免形状为 (2000,1)）
y = y.flatten()

# 获取排序索引
sorted_indices = np.argsort(y)

# 应用排序
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

# 保存排序后的数据
np.savez('robot_data_sorted.npz', X=X_sorted, y=y_sorted)
print("数据已按 label 排序并保存为 'robot_data_sorted.npz'")
