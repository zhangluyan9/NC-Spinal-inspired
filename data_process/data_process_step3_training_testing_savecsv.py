import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import random
import pandas as pd
import os

# 锁定随机种子（推荐三者都设）
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# -------------------------
# 1. 加载数据并预处理
# -------------------------
data = np.load('robot_data_simulated_function_low_original_threedimension.npz')
X = data['X']  # (2000, 50, 9)
y = data['y']  # (2000,)
print(X.shape)

# -------------------------
# 2. 保存X数据为CSV文件
# -------------------------
print("正在保存X数据为CSV文件...")

print(f"原始X形状: {X.shape}")  # (2000, 50, 9)

# 将NaN值替换为0
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"检测到 {nan_count:,} 个NaN值，将替换为0")
    X_clean = np.nan_to_num(X, nan=0.0)
    print(f"✅ NaN值已替换为0")
else:
    print(f"未检测到NaN值")
    X_clean = X.copy()

# 创建列名：sample_id + 9个特征列 + label列
feature_columns = [f'feature_{i}' for i in range(9)]
all_columns = ['sample_id'] + feature_columns + ['label']

# 创建用于存储所有数据的列表
all_rows = []

print(f"正在处理 {X_clean.shape[0]} 个样本...")

for sample_idx in range(X_clean.shape[0]):
    # 当前样本的50x9数据
    sample_data = X_clean[sample_idx]  # shape: (50, 9)
    current_label = y[sample_idx]  # 当前样本的标签
    
    # 为当前样本的50行添加sample_id和label
    for time_step in range(50):
        row = [sample_idx] + sample_data[time_step].tolist() + [current_label]
        all_rows.append(row)
    
    # 在样本之间添加空行（除了最后一个样本）
    if sample_idx < X_clean.shape[0] - 1:
        empty_row = [None] * len(all_columns)
        all_rows.append(empty_row)
    
    # 显示进度
    if (sample_idx + 1) % 200 == 0:
        print(f"   已处理 {sample_idx + 1}/{X_clean.shape[0]} 个样本")

# 创建DataFrame
print(f"创建DataFrame，总行数: {len(all_rows)}")
combined_df = pd.DataFrame(all_rows, columns=all_columns)

# 保存包含特征的CSV（不含标签）
X_df = combined_df.drop('label', axis=1)
csv_filename = 'robot_data_X_features.csv'
X_df.to_csv(csv_filename, index=False)
print(f"✅ X数据已保存为: {csv_filename}")
print(f"   文件大小: {os.path.getsize(csv_filename) / 1024 / 1024:.2f} MB")
print(f"   数据形状: {X_df.shape}")
print(f"   格式: 每个样本50行×9列，样本间有空行分隔")
print(f"   总行数: {X_clean.shape[0]} 样本 × 50 行 + {X_clean.shape[0]-1} 空行 = {len(all_rows)} 行")

# -------------------------
# 3. 保存y标签为CSV文件
# -------------------------
print("\n正在保存y标签为CSV文件...")

# 创建标签DataFrame（传统格式，每个样本一行）
y_df = pd.DataFrame({
    'sample_id': range(len(y)),
    'label': y
})

# 保存标签
y_csv_filename = 'robot_data_y_labels.csv'
y_df.to_csv(y_csv_filename, index=False)
print(f"✅ y标签已保存为: {y_csv_filename}")
print(f"   格式: 传统格式，每个样本一行")

# -------------------------
# 4. 保存合并的数据文件
# -------------------------
print("\n正在保存合并的数据文件...")

# 保存合并文件（直接使用already包含label的combined_df）
combined_csv_filename = 'robot_data_complete.csv'
combined_df.to_csv(combined_csv_filename, index=False)
print(f"✅ 完整数据已保存为: {combined_csv_filename}")
print(f"   包含: {len(feature_columns)} 个特征列 + 1个标签列 + 1个样本ID列")
print(f"   格式: 垂直堆叠的50×9矩阵，每个样本包含50行数据")

# -------------------------
# 5. 数据摘要信息
# -------------------------
print(f"\n📊 数据摘要:")
print(f"   原始X形状: {X.shape}")
print(f"   重塑后形状: {X_clean.shape}")
print(f"   标签形状: {y.shape}")
print(f"   唯一标签数: {len(np.unique(y))}")
print(f"   标签范围: {y.min()} 到 {y.max()}")

# 显示前几行数据预览
print(f"\n📋 数据预览 (前5行, 前10列):")
print(combined_df.iloc[:5, :11].to_string())

print(f"\n💾 生成的CSV文件:")
print(f"   1. {csv_filename} - 仅特征数据")
print(f"   2. {y_csv_filename} - 仅标签数据") 
print(f"   3. {combined_csv_filename} - 完整数据 (推荐使用)")

# -------------------------
# 6. 验证保存的数据
# -------------------------
print(f"\n🔍 验证保存的数据...")

# 重新加载并验证
loaded_df = pd.read_csv(combined_csv_filename)
print(f"   重新加载的数据形状: {loaded_df.shape}")

# 由于数据格式已改变（垂直堆叠的50x9矩阵），需要重新构建原始数据进行比较
print(f"   正在重构数据以进行验证...")

# 过滤掉空行
non_empty_df = loaded_df.dropna()
print(f"   去除空行后形状: {non_empty_df.shape}")

# 重构X数据
loaded_X_list = []
loaded_y_list = []

for sample_id in range(2000):
    # 获取当前样本的数据（非空行）
    sample_rows = non_empty_df[non_empty_df['sample_id'] == sample_id]
    
    if len(sample_rows) == 50:  # 确保有50行时间步数据
        # 提取特征列（排除sample_id和label列）
        features = sample_rows.iloc[:, 1:-1].values  # shape: (50, 9)
        loaded_X_list.append(features)
        
        # 获取标签（取第一行的标签值，因为同一个样本的所有行标签相同）
        label = sample_rows.iloc[0]['label']
        loaded_y_list.append(label)
    else:
        print(f"   ⚠️ 样本 {sample_id} 数据行数不正确: {len(sample_rows)}")
        # 调试信息：显示前几个样本的详细情况
        if sample_id < 5:
            print(f"      样本 {sample_id} 的行索引: {sample_rows.index.tolist()}")

# 调试信息：检查总的非空行数
expected_total_rows = 2000 * 50  # 每个样本50行
actual_total_rows = len(non_empty_df)
print(f"   期望的非空行数: {expected_total_rows}")
print(f"   实际的非空行数: {actual_total_rows}")

if len(loaded_X_list) == 2000:
    # 重构为numpy数组
    loaded_X_reconstructed = np.array(loaded_X_list)  # shape: (2000, 50, 9)
    loaded_y_reconstructed = np.array(loaded_y_list)  # shape: (2000,)
    
    print(f"   重构的X形状: {loaded_X_reconstructed.shape}")
    print(f"   重构的y形状: {loaded_y_reconstructed.shape}")
    
    # 验证形状
    shape_match = X_clean.shape == loaded_X_reconstructed.shape
    y_shape_match = y.shape == loaded_y_reconstructed.shape
    
    print(f"   X形状匹配: {'✅ 通过' if shape_match else '❌ 失败'}")
    print(f"   y形状匹配: {'✅ 通过' if y_shape_match else '❌ 失败'}")
    
    # 检查y数据
    y_match = np.array_equal(y, loaded_y_reconstructed)
    print(f"   y数据一致性: {'✅ 通过' if y_match else '❌ 失败'}")
    
    # 检查X数据
    if shape_match:
        X_match = np.allclose(X_clean, loaded_X_reconstructed)
        X_exact = np.array_equal(X_clean, loaded_X_reconstructed)
        
        print(f"   X数据一致性(近似): {'✅ 通过' if X_match else '❌ 失败'}")
        print(f"   X数据一致性(精确): {'✅ 通过' if X_exact else '❌ 失败'}")
        
        # 检查是否还有NaN值
        loaded_nan_count = np.isnan(loaded_X_reconstructed).sum()
        print(f"   加载数据中NaN数量: {loaded_nan_count}")
        
        if not X_match:
            diff = np.abs(X_clean - loaded_X_reconstructed)
            max_diff = diff.max()
            print(f"   最大差异: {max_diff:.2e}")
    else:
        X_match = False
        
    # 总体验证结果
    if X_match and y_match:
        print(f"\n🎉 数据保存成功！所有验证通过。")
        print(f"   ✅ CSV数据与原始数据完全一致")
        print(f"   ✅ NaN值已成功替换为0")
        print(f"   ✅ 50×9矩阵格式正确保存")
    else:
        print(f"\n⚠️  数据验证失败，请检查保存过程。")
        if not X_match:
            print(f"   ❌ X数据不一致")
        if not y_match:
            print(f"   ❌ y数据不一致")
else:
    print(f"\n❌ 数据重构失败，样本数量不正确: {len(loaded_X_list)}")
    print(f"   期望: 2000 个样本")
    print(f"   实际: {len(loaded_X_list)} 个样本")