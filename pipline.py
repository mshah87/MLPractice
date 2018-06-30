#import iris dataset
from sklearn import datasets
iris= datasets.load_iris()

'''
think of classifier as function 
x is input (features) and y is output (label)
'''
x= iris.data
y= iris.target

#partition to train and test
from sklearn.cross_validation import train_test_split

#xtrain and ytrain are features and labels for training set 
#xtest and ytest are features and labels for training set
# testSize= 0.5 means use half of data for testing
#essentially, 75 train and 75 test
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size = .5)



#classifier 1
from sklearn import tree

classifier= tree.DecisionTreeClassifier()

#classifier 2
'''
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier()
'''

#train classifier using training data
classifier.fit(xtrain, ytrain)

#classify testing data
predicitions= classifier.predict(xtest)

print predicitions

#calculate accuracy by comparing predicted labels
#to the true labels (ytest). then sum up score
from sklearn.metrics import accuracy_score
print accuracy_score(ytest, predicitions)

#learning data means to use training data to adjust parameters of model