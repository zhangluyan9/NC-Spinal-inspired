import numpy as np
import pandas as pd

# Load projected feature data
data = np.load('robot_data_sorted.npz')  # Your already-projected file
X = data['X']  # shape: (2000, 50, 6)
y = data['y']  # shape: (2000, 1)
# Load temperature maps for label 8 and 10
temp_08 = pd.read_csv('08.csv', header=None) # (50, 3)
temp_10 = pd.read_csv('10.csv', header=None) # (50, 3)
temp_08 = temp_08.iloc[:, 1:].values  # 转为 NumPy 数组，去掉第一列
temp_10 = temp_10.iloc[:, 1:].values  # 转为 NumPy 数组，去掉第一列

X_expanded = np.zeros((X.shape[0], X.shape[1], 9), dtype=float)
index_8_i=0
index_8_j=0
index_10_i=0
index_10_j=0

for i in range(X.shape[0]):

    label = y[i]
    temps = np.full((50, 3), 6000.0)
    # Fill expanded vector [ef1, fp1, t1, ef2, fp2, t2, ef3, fp3, t3]
    X_expanded[i, :, 0] = X[i, :, 0]  # eucl force 1
    X_expanded[i, :, 1] = X[i, :, 3]  # finger1_pos 
    X_expanded[i, :, 2] = temps[:, 0]

    X_expanded[i, :, 3] = X[i, :, 1]  # eucl force 2
    X_expanded[i, :, 4] = X[i, :, 4]  # finger2_pos
    X_expanded[i, :, 5] = temps[:, 1]
    

    X_expanded[i, :, 6] = X[i, :, 2]  # eucl force 3
    X_expanded[i, :, 7] = X[i, :, 5]  # finger3_pos
    X_expanded[i, :, 8] = temps[:, 2]
    #print(i,label)



print(X_expanded.shape)
#print(X_expanded[800])
# Save to new file
np.savez('robot_data_expanded_with_temp_force_angel.npz', X=X_expanded, y=y)
