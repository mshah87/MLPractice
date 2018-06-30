from scipy.spatial import distance

#a is point in training data
#b is point in testing data 

def euc(a,b):
    return distance.euclidean(a,b)

#implement class for classifier
class ScrappyKNN(): 

    #training
    def fit(self, xtrain, ytrain):
        self.xtrain= xtrain
        self.ytrain= ytrain

    #prediction
    #returns predictions for labels
    def predict(self, xtest):
        #gotta return list of predictions
        predictions= []
        #each row contains features for testing example
        for row in xtest:
            
            #method "closest" finds closest training point to test point
            label= self.closest(row)
            predictions.append(label)

        return predictions

        '''
        to make prediction for test point, we will calculate distance of all training points. then predict that testing point has same label as closest one.
        K=1, so this is a nearest neighbor classifier 

        to find nearest neighbor, measure straight line distance between two points. Use euclidean distance (a^2+b^2 =c^2). This is for 2d space if u have 2 features in dataset. 
        If you had 3 features, so 3d, then we would be in a cube. What if we have 4 features like iris dataset.  
        More features means more terms in equation
        '''

    def closest(self, row):
        #calculate distance from test point to first training point
        #keep track of shortest distance we found so far
        shortdist= euc(row, self.xtrain[0])  
        #keep track of index of training point thats closest
        index= 0

        #iterate over all other training points 
        for i in range(1, len(self.xtrain)):
            dist= euc(row, self.xtrain[i])
            #everytime we find a closer training point, we make that our new shortdist
            if dist < shortdist:
                shortdist= dist
                index= i
        return self.ytrain[index]
            
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


#classifier 2
#from sklearn.neighbors import KNeighborsClassifier
#we making own classifier
classifier= ScrappyKNN()

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