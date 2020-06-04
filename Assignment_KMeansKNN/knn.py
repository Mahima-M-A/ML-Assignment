import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# function to load data from the file and preprocessing 
def loadData(lines):
    rows =[]
    headers = lines[0].strip().split(',')
    #cast each row's values from string to float and form a list of rows
    for line in lines[1:]:
        flag = 0
        l = []
        for x in line.split(','):
            if(x == ''):
                flag=1
                break
            else:
                l.append(float(x))
        if flag == 0:
            rows.append(l)

    columns = len(rows[0]) - 1 #no. of features
    print('No. of records: ',len(rows))
    return headers, rows, columns

# to have the same shuffle and split every time the code is run
def shuffleData(rows):
    np.random.seed(0)
    np.random.shuffle(rows)
    return rows

#to split data
def splitData(rows):
    trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = [], [], [], [], [], []

    #70% of the data is used as train data
    for i in range(int(0.7 * len(rows))):
        trainData.append(rows[i])
        trainTarget.append(rows[i][-1]) #the output column values of train data
        trainFeatures.append(rows[i][:-1]) #the list of feature values of train data

    #remainder(30%) of the data is used as test data 
    for i in range(int(0.7 * len(rows)), len(rows)):
        testData.append(rows[i])
        testTarget.append(rows[i][-1]) #the output column values of test data
        testFeatures.append(rows[i][:-1]) #the list of feature values of test data

    #Display train data and test data target values
    print('trainTarget: ',  trainTarget)
    print('testTarget: ', testTarget)
    return trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures

#distance metrics
def euclideanDistance(v1, v2):
    dist = 0
    for i in range(len(v1) - 1):
        dist += (v1[i] - v2[i])**2
    return math.sqrt(dist)

def manhattanDistance(v1, v2):
    dist = 0
    for i in range(len(v1) - 1):
        dist += abs(v1[i] - v2[i])
    return dist

def chebyshevDistance(v1, v2):
    dist = []
    for i in range(len(v1) - 1):
        dist.append(abs(v1[i] - v2[i]))
    return max(dist)

#to calculate the distance between the test data pt with every train data pt
def calcDist(testRow, distFunc, trainFeatures):
    distances = []
    for index,trainRow in enumerate(trainFeatures):
        dis = distFunc(trainRow, testRow)
        distances.append((index, dis))
    return sorted(distances, key = lambda x: x[1])

#to find the best label for the data point
def mode(labels):
    return Counter(labels).most_common(1)[0][0]

#knn model
def knnModel(kValue, testFeatures, trainFeatures, distFunc):
    kNearestNeighbours = []
    kNearestLabels = []
    for index,row in enumerate(testFeatures):
        neighbours = calcDist(row, distFunc, trainFeatures)
        kNearestNeighbours.append(neighbours[:kValue])
        klabels = [trainTarget[i] for i, dis in kNearestNeighbours[index]]
        nearestLabel = mode(klabels)
        kNearestLabels.append((index, nearestLabel))
    return kNearestLabels

#normalizing the dataset
def normalizeDataset():
    column, normalRow, normalRows = [], [], []
    for i in range(columns+1):
        cols=[j[i] for j in rows]
        column.append(cols)
        cols=[]

    minimum=[min(i) for i in column]
    maximum=[max(i) for i in column]
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            if not j == len(rows[0])-1:
                normalRow.append((rows[i][j] - minimum[j]) / (maximum[j] - minimum[j]))
            else:
                normalRow.append(rows[i][j])
        normalRows.append(normalRow)
        normalRow = []

    return normalRows

#performance metrics calculation
def classificationReport(kNearestLabels):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, label in kNearestLabels:
        if testTarget[i] == label and label == 1:
            TP += 1
        elif testTarget[i] == label and label == 0:
            TN += 1
        elif testTarget[i] != label and label == 1:
            FP += 1
        else:
            FN += 1
    
    confusionMatrix = [[TP, FN], [FP, TN]]
    print('Confusion Matrix:\n', str(confusionMatrix[0][0])+' '+str(confusionMatrix[0][1])+'\n '+str(confusionMatrix[1][0])+' '+str(confusionMatrix[1][1]))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: ',Accuracy)
    Precision = TP / (TP + FP)
    print('Precision: ',Precision)
    Recall = TP / (TP + FN)
    print('Recall: ',Recall)
    return Accuracy

fileName = "diabetes.csv"
fp = open(fileName, 'r')
lines = fp.readlines()

headers, rows, columns = loadData(lines) #function call to load data (also acts as the main function)

rows = shuffleData(rows) #function call to shuffle data

trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = splitData(rows) #function call to split data

#performance of the model on train data
kNearestLabels = knnModel(13, trainFeatures, trainFeatures, distFunc=euclideanDistance)
scores = []
for i, label in kNearestLabels:
    correct = 0
    if trainTarget[i] == label:
        correct += 1
    scores.append(correct)
print('\nAccuracy on training data: ', (sum(scores)/float(len(scores))))

#performance of the model at different k values
accuracy = []
for kValue in range(1, 16): 
    kNearestLabels = knnModel(kValue, testFeatures, trainFeatures, distFunc=euclideanDistance)
    print('\nPerformance of KNN model using euclidean distance when k = '+str(kValue))
    accuracy.append(classificationReport(kNearestLabels))

#plot to show the variation of accuracy with different k values
plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("k values")
plt.show()

#choosing the k value that gave the highest accuracy
kValue = 13

#using different distance metrics: manhattanDistance and chebyshevDistance
distanceMetrics = [euclideanDistance, manhattanDistance, chebyshevDistance]
for metric in distanceMetrics:
    kNearestLabels = knnModel(kValue, testFeatures, trainFeatures, distFunc=metric)
    print('\nPerformance of KNN model using '+ metric.__name__ + ' when k = '+str(kValue))
    classificationReport(kNearestLabels)

#Normalizing the dataset
normalRows = normalizeDataset()
#splitting normalized dataset
normalTrainData, normalTestData, normalTrainTarget, normalTestTarget, normalTrainFeatures, normalTestFeatures = splitData(normalRows)

#performance of the KNN model on the normalized dataset 
print("\nPerformance on normalizing the dataset")
kNearestLabels = knnModel(kValue, normalTestFeatures, normalTrainFeatures, distFunc=euclideanDistance)
classificationReport(kNearestLabels)

#ablation study on the normalized dataset
# converting list of lists to a dataframe
df = pd.DataFrame(normalRows, columns = headers)

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(25,35))
#plot heat map
ax=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="Blues")
plt.show()

#dropping the columns that are least correlated with the outcome
df.drop(['BloodPressure', 'SkinThickness'], inplace=True, axis=1)
print(df.head())
newRows = df.values.tolist()

newRows = shuffleData(newRows) #function call to shuffle data

trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = splitData(newRows) #function call to split data
kNearestLabels = knnModel(kValue, testFeatures, trainFeatures, distFunc=euclideanDistance)
print("\nPerformance of the model after dropping the least correlated columns from the normalized dataset")
classificationReport(kNearestLabels)