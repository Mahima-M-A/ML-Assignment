import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#activation function to return a value 0 if sum<0 and 1 if sum>=0 (sigmoid function, tanh etc can also be used)
def predict(trainDataRow):
    sum = 0
    for i in range(columns + 1):
        sum += trainDataRow[i] * weights[i+1]
    sum += weights[0]
    return 0 if sum < 0 else 1

#plot showing error(misclassified) vs epochs
def plotError(epochs):
    plt.plot(np.arange(1, epochs+1), errorValues)
    plt.xlabel('epochs')
    plt.ylabel('misclassifications')
    plt.show()

#display function
def display(weights, epochs, learningRate):
    print("\n\nNo. of epochs: ", epochs)
    print("Learning rate: ", learningRate)
    print("Learnt weights: ", weights[1:])
    plotError(epochs)

#function to train weights
def learnWeights(epochs, learningRate, weights, columns):
    for epoch in range(epochs):
        misclassified = 0
        for j in range(len(trainFeatures)):
            p = predict(trainFeatures[j])
            error = trainTarget[j] - p

            if(error):
                misclassified += 1

            weights[0] += learningRate * error

            for k in range(1, columns+1):
                weights[k] += learningRate * error * trainFeatures[j][k-1]

        errorValues.append(misclassified)
        print("epoch: "+str(epoch)+"  updated weights: "+str(weights))

    display(weights, epochs, learningRate) #function call to display learnt weights, epochs and learning rate

#to initialize epochs, learningRate and weights
def initialize(columns):
    epochs = 1000
    learningRate = 0.01
    weights.append(1.0)

    for _ in range(1, columns+1):
        r = np.random.uniform(0.0, 1.0) #assigns random float values in the range 0.0 to 0.1
        weights.append(r)

    print("Initial weights: ", weights[1:])

    learnWeights(epochs, learningRate, weights, columns) #function call to train weights

#to split data
def splitData(rows):
    # to have the same shuffle and split every time the code is run
    np.random.seed(0)
    np.random.shuffle(rows)

    #70% of the data is used as train data
    for i in range(int(0.7 * len(rows))):
        trainData.append(rows[i])
        trainTarget.append(int(rows[i][-1])) #the output column values of train data
        trainFeatures.append(rows[i][:-1]) #the list of feature values of train data

    #remainder(30%) of the data is used as test data
    for i in range(int(0.3 * len(rows))):
        testData.append(rows[i])
        testTarget.append(int(rows[i][-1])) #the output column values of test data
        testFeatures.append(rows[i][:-1]) #the list of feature values of test data

    #Display total no. of records, train data and test data target values
    print('No. of records: ',len(trainData) + len(testData))
    print('trainTarget: ',  trainTarget)
    print('testTarget: ', testTarget)

# function to load data from the file
def loadData():
    fp = open(fileName, 'r')
    lines = fp.readlines()

    #cast each row's values from string to float and form a list of rows
    for line in lines:
        l = [float(x) for x in line.split('\t')]
        rows.append(l)

    columns = len(rows[0]) - 1 #no. of features

    splitData(rows) #function call to split data
    initialize(columns) #function call to initialize epochs, learning rate and weights

#to calc accuracy metrics
def calcAccMetrics():
    TP, TN, FP, FN = 0, 0, 0, 0 #true positive, true negative, false positive, false negative(accuracy metrics)
    for i in range(len(testData)):
        if testTarget[i] == predictedOutput[i] and predictedOutput[i] == 1:
            TP += 1
        elif testTarget[i] == predictedOutput[i] and predictedOutput[i] == 0:
            TN += 1
        elif testTarget[i] != predictedOutput[i] and predictedOutput[i] == 1:
            FP += 1
        else:
            FN += 1

    confusionMatrix = [[TP, FN], [FP, TN]]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return confusionMatrix, Accuracy, Precision, Recall

#declaration of the required variables
fileName = 'data.txt' #file name
rows = []
weights = []
columns = 0
epochs = 0
learningRate = 0
trainData, testData, trainTarget, testTarget, trainFeatures, testFeatures = [], [], [], [], [], []
predictedOutput = []
errorValues = []


loadData() #function call to load data (also acts as the main function)

scores = []

#Accuracy of model on training data
for i in range(len(trainFeatures)):
    correct = 0
    if trainTarget[i] == predict(trainFeatures[i]):
        correct += 1
    scores.append(correct / float(len(trainTarget)) * 100.0)
print('Accuracy on training data:', (sum(scores) / float(len(scores))) * 100, '%')

#testing the perceptron using test data features
print('Actual target values - Predicted target values')
for i in range(len(testFeatures)):
    predictedOutput.append(predict(testFeatures[i]))
    print('     ',testTarget[i],'   -   ',predictedOutput[i])

confusionMatrix, Accuracy, Precision, Recall = calcAccMetrics() #function call to calculate the accuracy metrics

print('Confusion Matrix:\n', str(confusionMatrix[0][0])+' '+str(confusionMatrix[0][1])+'\n '+str(confusionMatrix[1][0])+' '+str(confusionMatrix[1][1]))
print('Accuracy: ',Accuracy)
print('Precision: ',Precision)
print('Recall: ',Recall)
