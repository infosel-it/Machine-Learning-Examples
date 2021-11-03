from numpy.linalg import eig

covmatrix = [[3,4],[4,3]]

eigenval, eigenvector = eig(covmatrix)

print("Eigen Value:", eigenval)
print("Eigen Vector:\n", eigenvector)