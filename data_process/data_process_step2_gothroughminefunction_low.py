import numpy as np
import pandas as pd
from ISI_IBI_MBI_low_p import *
# Load projected feature data
#data = np.load('robot_data_expanded_with_temp.npz')  # Your already-projected file
# data = np.load('robot_data_expanded_with_temp_force.npz')  # Your already-projected file
# data = np.load('robot_data_expanded_with_temp_angel.npz')  # Your already-projected file
# data = np.load('robot_data_expanded_with_temp_temper.npz')  # Your already-projected file
data = np.load('robot_data_expanded_with_temp_force_angel.npz')  # Your already-projected file

X = data['X']  # shape: (2000, 50, 9)
y = data['y']  # shape: (2000, 1)
print(X[0][46])

print(X.shape)
print(y.shape)
X_out = np.zeros_like(X)


for i in range(X.shape[0]):
    print(i)
    for j in range(X.shape[1]):
        # 每次取三个维度
        for k in range(0, 9, 3):
            x1, x2, x3 = X[i, j, k:k+3]
            y1, y2, y3 = simulate_from_RLs(x3, x1, x2)
            print(y1,y2,y3)
            X_out[i, j, k:k+3] = [y1, y2, y3]

# 保存处理后的结果
#np.savez('robot_data_simulated_function_low_original_threedimension.npz', X=X_out, y=data['y'])
#np.savez('robot_data_simulated_function_low_original_threedimension.npz', X=X_out, y=data['y'])
np.savez('robot_data_simulated_function_low_original_threedimension_force_angel.npz', X=X_out, y=data['y'])
# np.savez('robot_data_simulated_function_low_original_threedimension_angel.npz', X=X_out, y=data['y'])
#np.savez('robot_data_simulated_function_low_original_threedimension_temper.npz', X=X_out, y=data['y'])
print(X.shape)

