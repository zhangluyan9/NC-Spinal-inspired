import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import random

seed = 80
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# -------------------------
# 1. 加载数据并预处理
# -------------------------
# np.savez('robot_data_simulated_function_low_original_threedimension.npz', X=X_out, y=data['y'])
# np.savez('robot_data_simulated_function_low_original_threedimension_force.npz', X=X_out, y=data['y'])
# np.savez('robot_data_simulated_function_low_original_threedimension_angel.npz', X=X_out, y=data['y'])
#np.savez('robot_data_simulated_function_low_original_threedimension_temper.npz', X=X_out, y=data['y'])

data = np.load('robot_data_simulated_function_low_original_threedimension_temper.npz')
X = data['X']  # (2000, 50, 9)
y = data['y']  # (2000,)

X = np.nan_to_num(X, nan=0.0)
X = X.reshape(X.shape[0], -1)  # (2000, 450)

# -------------------------
# 2. 按类别随机划分为 80% 10% 10%
# -------------------------
num_classes = 20
samples_per_class = 100
train_idx_list, val_idx_list, test_idx_list = [], [], []

for cls in range(num_classes):
    class_indices = np.where(y == cls)[0]
    np.random.shuffle(class_indices)
    
    n_train = int(0.7 * samples_per_class)   # 80
    n_val   = int(0.2 * samples_per_class)   # 10
    n_test  = samples_per_class - n_train - n_val  # 10
    #print(n_train,n_val)
    train_idx_list.append(class_indices[:n_train])
    val_idx_list.append(class_indices[n_train:n_train+n_val])
    test_idx_list.append(class_indices[n_train+n_val:])
#print(val_idx_list)
train_idxs = np.concatenate(train_idx_list)
val_idxs   = np.concatenate(val_idx_list)
test_idxs  = np.concatenate(test_idx_list)

# -------------------------
# 3. 提取并转为 Tensor
# -------------------------
X_train = torch.from_numpy(X[train_idxs]).float()
y_train = torch.from_numpy(y[train_idxs]).long()
X_val   = torch.from_numpy(X[val_idxs]).float()
y_val   = torch.from_numpy(y[val_idxs]).long()
X_test  = torch.from_numpy(X[test_idxs]).float()
y_test  = torch.from_numpy(y[test_idxs]).long()

# -------------------------
# 4. 构造 DataLoader
# -------------------------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# -------------------------
# 5. 打印维度确认
# -------------------------
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

# -------------------------
# 6. 保存 Tensor 数据
# -------------------------
torch.save({'X': X_train, 'y': y_train}, 'train_data_3dimension_temper.pt')
torch.save({'X': X_val,   'y': y_val},   'val_data_3dimension_temper.pt')
torch.save({'X': X_test,  'y': y_test},  'test_data_3dimension_temper.pt')

print("Datasets saved as train_data_3dimension_temper.pt, val_data_3dimension_temper.pt, test_data_3dimension_temper.pt")
