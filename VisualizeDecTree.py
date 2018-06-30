import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris= load_iris()

'''
#names of features
print iris.feature_names
#names of diff types of flowers
print iris.target_names

#measurements of flower (prints out measuements of first flower)
print iris.data[0]

#labels (prints out first flower)
print iris.target[0]

#iterate over entire dataset and print it out
for i in range(len(iris.target)):
    print "example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

    '''
#split up data
#remove one example of each flower and put aside for later
#these removed examples are called testing data
#keep seperate from training data
removeOne= [0, 50, 100]


#training data
'''
the rows that are deleted are the rows which are indexed 
by the values in removeOne vertically. 

if removeOne = [3,4,5], and axis = 0, the 3rd,
 4th and 5th rows would be deleted.

if test_idx=[3,4,5], but axis = 1, then the 3rd, 4th and 
5th columns are deleted, as axis=1 is the horizontal axis.
'''
trainTarget= np.delete(iris.target, removeOne)
trainData= np.delete(iris.data, removeOne, axis=0)

#testing data
#only examples that were removed
testTarget= iris.target[removeOne]
testData= iris.data[removeOne]

#create decision tree classifier 
#train it on training data
classifier= tree.DecisionTreeClassifier()
classifier.fit(trainData, trainTarget)

print testTarget
print classifier.predict(testData)