from sklearn import tree


#training data
# weight | texture | label
# 150      bumpy      orange
# 170      bumpy      orange
# 140      smooth     apple
# 130      smooth     apple

#first two columns
# 0 for bumpy, 1 for smooth
features  = [[140,1],[130,1],[150,0],[170,0]]

#0 for apple, 1 for orange 
labels= [0,0,1,1]

#right now its empty bag of rules
#gotta train it with learning algorithm
classifier= tree.DecisionTreeClassifier()

#training algorithm 
#fit means find patterns in data 
classifier= classifier.fit(features, labels)

print classifier.predict([[160, 0]])