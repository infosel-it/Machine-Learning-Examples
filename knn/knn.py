
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

cgpa=[9.2,8,8.5,6,6.5,8.2,5.8,8.9]

assesment=[85,80,81,45,50,72,38,91]

projectsubmit=[8,7,8,5,4,7,5,9]

result =['Pass','Pass','Pass','Fail','Fail','Pass','Fail','Pass']

print("Len of cgpa:",len(cgpa))
print("Len of assesment:",len(assesment))
print("Len of projectsubmit:",len(projectsubmit))
print("Len of result:",len(result))     


result=le.fit_transform(result)
print("Result Encoded", result)

features=list(zip(cgpa,assesment,projectsubmit))
print("Features", features)
print("\n")

print("-----------K = 3 -------------")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,result)

predicted= model.predict([[6.1,40,5]])
print("Predicted for K=3: (6.1,40,5) ", predicted)

predicted= model.predict([[8.9,91,9]])
print("Predicted: K=3 (8.9,91,9) ", predicted)

print("\n")

print("-----------K = 5 -------------")

model1 = KNeighborsClassifier(n_neighbors=5)  
model1.fit(features,result)

predicted1= model1.predict([[6.1,40,5]])
print("Predicted for K=5: (6.1,40,5)", predicted1)

predicted1= model1.predict([[8.9,91,9]])
print("Predicted: K=5 (8.9,91,9) ", predicted1)

print("\n")
print("-----------K = 7 -------------")

model1 = KNeighborsClassifier(n_neighbors=7)  
model1.fit(features,result)

predicted1= model1.predict([[6.1,40,5]])
print("Predicted for K=7: (6.1,40,5)", predicted1)

predicted1= model1.predict([[8.9,91,9]])
print("Predicted: K=7 (8.9,91,9) ", predicted1)

X_train, X_test, y_train, y_test = train_test_split(
            features, result, test_size = 0.2, random_state=42)

neighbors = np.arange(2, 7)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    print(" ---- Fitting for  Neihbors K ----:", k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)


print(" Train Accuracy :", train_accuracy)
print(" Test Accuracy :", test_accuracy)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()