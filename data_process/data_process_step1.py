import pandas as pd
import numpy as np

# 文件路径（你自己的 CSV 文件）
file_path = 'modified_robot_data.csv'  # 改成你自己的路径

# 总体结构说明：
# 每个样本占 52 行：
# - 第 0 行：只有 label，在第 1 列（即 column index = 1）
# - 第 1 行：表头（忽略）
# - 第 2~51 行（共 50 行）：有用的特征数据

# 数据总数
num_samples = 2000
rows_per_sample = 52
useful_rows = 50

# 列索引（0-based）
label_col_idx = 8
feature_col_idxs = [1, 2, 3, 4, 5, 6]  # [eucl force 1, eucl force 2, eucl force 3, finger1_pos, finger2_pos, finger3_pos]

# 读取 CSV，不跳过任何行或表头
df = pd.read_csv(file_path, header=None)

# 初始化输出数据结构
X = np.zeros((num_samples, useful_rows, len(feature_col_idxs)), dtype=float)
y = np.zeros((num_samples, 1), dtype=int)

# 遍历每个样本
for i in range(num_samples):
    base = i * rows_per_sample
    y[i, 0] = df.iloc[base, label_col_idx] # 从第 0 行第 1 列读取 label
    print(i,y[i, 0])
    X[i] = df.iloc[base + 2 : base + 2 + useful_rows, feature_col_idxs].astype(float).values  # 50 行 × 6 特征
    
# 保存为 .npz 文件
    
import numpy as np

# # Step 1: Load the dataset
# data = np.load('robot_data.npz')
# X = data['X']  # shape: (2000, 50, 6)
# y = data['y']

# Step 2: Reshape to (2000*50, 6) to compute min/max across all samples and time steps
X_flat = X.reshape(-1, X.shape[-1])  # shape: (100000, 6)

# Step 3: Compute per-column min and max
min_vals = X_flat.min(axis=0)
max_vals = X_flat.max(axis=0)

print(min_vals,max_vals)

# Step 4: Normalize each feature column to [0, 1]
X_norm = (X - min_vals) / (max_vals - min_vals)

# Step 5: Project normalized values to target ranges
#反过来，最小 10000，最大 4000
X_projected = np.empty_like(X_norm)
X_projected[:, :, 0:3] = 10000 + X_norm[:, :, 0:3] * (4000 - 10000)
#反过来，最小 15000，最大 5000
X_projected[:, :, 3:6] = 15000 + X_norm[:, :, 3:6] * (5000 - 15000)

# Step 6: Save the projected data
np.savez('robot_data_projected.npz', X=X_projected, y=y)

labels, counts = np.unique(y, return_counts=True)

# 打印结果
for label, count in zip(labels, counts):
    print(f"Label {label}: {count} samples")

