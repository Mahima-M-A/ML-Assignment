import numpy as np
from sklearn import multiclass
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# function to load data from the file
def loadData():
    fp = open(fileName, 'r')
    lines = fp.readlines()
    rows =[]
    headers = lines[0].strip().split(',')[1:]
    #cast each row's values from string to float and form a list of rows
    for line in lines[1:]:
        l = []
        for x in line.strip().split(',')[1:]:
            if x == 'setosa': l.append(0)
            elif x == 'versicolor': l.append(1)
            elif x == 'virginica': l.append(2)
            else: l.append(float(x))
        rows.append(l)

    columns = len(rows[0]) - 1 #no. of features

    return headers, rows, columns

# to have the same shuffle and split every time the code is run
def shuffleData():
    np.random.seed(0)
    np.random.shuffle(rows)
    return rows

#to split data
def splitData():
    trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = [], [], [], [], [], []

    #70% of the data is used as train data
    for i in range(int(0.7 * len(rows))):
        trainData.append(rows[i])
        trainTarget.append(rows[i][-1]) #the output column values of train data
        trainFeatures.append(rows[i][:-1]) #the list of feature values of train data

    #remainder(30%) of the data is used as test data 
    for i in range(int(0.3 * len(rows))):
        testData.append(rows[i])
        testTarget.append(rows[i][-1]) #the output column values of test data
        testFeatures.append(rows[i][:-1]) #the list of feature values of test data

    #Display total no. of records, train data and test data target values
    print('No. of records: ',len(trainData) + len(testData))
    print('trainTarget: ',  trainTarget)
    print('testTarget: ', testTarget)
    return trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures

# plot of every pair of features
def plotPairWise():
    sns.pairplot(df, hue="Species")
    plt.show()

# scatter plot
def scatterplotOfBestPair():
    ax = sns.scatterplot(x = 'Petal.Length', y = 'Petal.Width', hue = 'Species', data = df)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Scatter plot - petal length vs petal width')
    plt.show()

def myLinspace(minValue, maxValue, steps):
    difference = maxValue - minValue
    return np.linspace(minValue - 1.0 * difference, maxValue + 1.0 * difference, steps)

# plot ovo and ovr classifications
def ovo_ovr_plot():
    x = df.loc[:, 'Petal.Length':'Petal.Width'].values
    y = df['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values

    ovoClassifier = SVC(kernel='linear') # SVC follows one versus one approach
    ovrClassifier = LinearSVC() # LinearSVC follows one versus rest approach

    # accuracy of ovo classifier
    ovoClassifier.fit(trainFeatures, trainTarget)
    predictedOutputs = ovoClassifier.predict(testFeatures)
    accuracy = accuracy_score(testTarget, predictedOutputs)
    print('Accuracy of the ovo classification: ', accuracy * 100, '%')

    # accuracy of ovr classifier
    ovrClassifier.fit(trainFeatures, trainTarget)
    predictedOutputs = ovrClassifier.predict(testFeatures)
    accuracy = accuracy_score(testTarget, predictedOutputs)
    print('Accuracy of the ovr classification: ', accuracy * 100, '%')

    ovoClassifier.fit(x, y)
    ovrClassifier.fit(x, y)

    steps = 200
    x0 = myLinspace(min(x[:,0]), max(x[:,0]), steps)
    x1 = myLinspace(min(x[:,1]), max(x[:,1]), steps)
    xx0, xx1 = np.meshgrid(x0, x1)
    meshData = np.c_[xx0.ravel(), xx1.ravel()]
    ovoDecisionFunc = ovoClassifier.decision_function(meshData)
    ovrDecisionFunc = ovrClassifier.decision_function(meshData)

    newDecisionFunc = multiclass._ovr_decision_function(ovrDecisionFunc < 0, -ovrDecisionFunc, 3).reshape(steps, steps, 3)
    ovoDecisionFunc = ovoDecisionFunc.reshape(steps, steps, 3)
    ovrDecisionFunc = ovrDecisionFunc.reshape(steps, steps, 3)

    colors = ['red', 'green', 'blue']
    yColor = [colors[i] for i in y]

    plt.figure(figsize = (18, 12))
    contourColors = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]

    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.scatter(x[:,0], x[:,1], c = yColor)
        plt.contourf(xx0, xx1, ovoDecisionFunc[:,:,i], 20, cmap=contourColors[i], alpha=0.5) # plots ovo classifications
        plt.subplot(2, 3, i + 4)
        plt.scatter(x[:,0], x[:,1], c = yColor)
        plt.contourf(xx0, xx1, newDecisionFunc[:,:,i], 20, cmap=contourColors[i], alpha=0.5) # plots ovr classifications

    plt.show()

# test the model using different kernels
def useOfKernels():
    for kernel in ['linear', 'rbf', 'poly']:
        svmClassifier = SVC(kernel = kernel, degree = 2) if kernel == 'poly' else SVC(kernel = kernel) 
        svmClassifier.fit(trainFeatures, trainTarget)
        predictedOutputs = svmClassifier.predict(testFeatures)
        accuracy = accuracy_score(testTarget, predictedOutputs)
        print('Accuracy of SVM using ',kernel, 'kernel: ', accuracy * 100, '%')
        plot_confusion_matrix(svmClassifier, testFeatures, testTarget)
        plt.show()
        print(classification_report(testTarget, predictedOutputs))


fileName = 'iris.csv' #file name

headers, rows, columns = loadData() #function call to load data (also acts as the main function)

rows = shuffleData() #function call to shuffle data

trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = splitData() #function call to split data

# SVM model
svm_model = SVC(gamma = 'scale') # default uses rbf kernel

# training the model with the training data
svm_model.fit(trainFeatures, trainTarget)

# testing the model with the training data
predictedOutput = svm_model.predict(trainFeatures)

# calculating the model accuracy on training data
trainAccuracy = accuracy_score(trainTarget, predictedOutput)
print('Accuracy of the model: ', trainAccuracy * 100, '%')

# converting list of lists to a dataframe
df = pd.DataFrame(rows, columns = headers)

# mapping values assigned to the species names back to names 
df['Species'] = df['Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df) # modified dataframe 

# visualization of every feature in the dataset with every other using pairplot 
plotPairWise()

scatterplotOfBestPair() # scatter plot of the best pair

ovo_ovr_plot() # function call to plot ovo and ovr classifications

useOfKernels() # test model using different kernels