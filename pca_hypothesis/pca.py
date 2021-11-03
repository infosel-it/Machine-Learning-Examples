import numpy as np
from sklearn import decomposition

X =np.array([[1,3],[4,3]])

print("Original Matrix :\n",X)
print("Original Shape :",X.shape)

print("\n")
print(" ---- Components = 2 ----------")
pca = decomposition.PCA(n_components=2)
x_pca = pca.fit_transform(X)
print("Transformed Matrix :\n",x_pca)
print("Transformed Matrix Shape :\n", x_pca.shape)

xinverse = pca.inverse_transform(x_pca)
print("Inverse Matrix :\n", xinverse)

print("Inverse Matrix Shape :", xinverse.shape)
print("\n")

print("Explained Variance :", pca.explained_variance_)
print("Explained Value :", pca.singular_values_)

print("\n")
print(" ---- Components = 1 ----------")
pca = decomposition.PCA(n_components=1)
x_pca = pca.fit_transform(X)
print("Transformed Matrix :\n",x_pca)
print("Transformed Matrix Shape :\n", x_pca.shape)

xinverse = pca.inverse_transform(x_pca)
print("Inverse Matrix :\n", xinverse)

print("Inverse Matrix Shape :", xinverse.shape)
print("\n")

print("Explained Variance :", pca.explained_variance_)
print("Explained Value :", pca.singular_values_)