with open('/Users/yzh/Desktop/DataM/msdchallenge/result', 'w') as file:
    file.write(str( "result not written") )
import math
import os
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext("local", "Simple App")
trainPaths = ['/Users/yzh/Desktop/DataM/msdchallenge/EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt',
'/Users/yzh/Desktop/DataM/msdchallenge/EvalDataYear1MSDWebsite/year1_test_triplets_visible.txt'
,'/Users/yzh/Desktop/DataM/msdchallenge/EvalDataYear1MSDWebsite/year1_valid_triplets_visible.txt']
testPath="/Users/yzh/Desktop/DataM/msdchallenge/EvalDataYear1MSDWebsite/year1_valid_triplets_hidden.txt"


class Maps():
  def __init__(self):
      self.songIntMap = {}
      self.userIntMap = {}
  def toDict(self, path, numLines):
    """
    user1: 0
    user2: 1
    ...
    """
    with open(path, 'r') as file:
        lines=[next(file) for x in range(numLines)]
    # print(lines[0], lines[-1], len(lines))
    for line in lines:
        contents = line.split('\t')
        user = contents[0]
        song = contents[1]
        if user not in self.songIntMap:
            self.userIntMap[ user ] = self.userIntMap.__len__()
        
        if song not in self.songIntMap:
            self.songIntMap[ song ] = self.songIntMap.__len__()

maps=Maps()

#loading test
maps.toDict(trainPaths[0], 10000)
maps.toDict(trainPaths[1], 10000)
maps.toDict(trainPaths[2], 2000)
userIntMap= sc.broadcast(maps.userIntMap) #broadcast stringId: int map to all worker nodes
songIntMap=sc.broadcast(maps.songIntMap)


#train
r1 = sc.textFile(trainPaths[0]).zipWithIndex().filter(lambda vi: vi[1] < 10000).keys()
r2 = sc.textFile(trainPaths[1]).zipWithIndex().filter(lambda vi: vi[1] < 10000).keys()
r3 = sc.textFile(trainPaths[2]).zipWithIndex().filter(lambda vi: vi[1] < 2000).keys()

data = sc.union([r1, r2, r3])
# Load and parse the data


#may need to broadcast for performance
ratings = data.map(lambda l: l.split('\t')).map(
    lambda l: [ userIntMap.value[ l[0] ] , songIntMap.value[ l[1] ] , l[2] ])\
    .map(lambda l: Rating(int( l[0]  ), int( l[1] ), float(l[2])) )

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)


# Evaluate the model on training data
traindata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(traindata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()




# Evaluate the model on test data
testRatings = sc.textFile(testPath).map(lambda l: l.split('\t')).filter( #only calculate RMSE for met entries
    lambda test: test[0] in userIntMap.value and test[1] in songIntMap.value
).map(
    lambda l: [ userIntMap.value[ l[0] ] , songIntMap.value[ l[1] ] , l[2] ])\
    .map(lambda l: Rating(int( l[0]  ), int( l[1] ), float(l[2])) )
testdata = testRatings.map(lambda p: (p[0], p[1]))
testCount=testRatings.count()
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = testRatings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
testMSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()






with open('/Users/yzh/Desktop/DataM/msdchallenge/result', 'w') as file:
    file.write(str( math.sqrt(MSE))+','+str(math.sqrt(testMSE)) +','+str( testCount) )
print("Mean Squared Error = " + str(MSE))

# Save and load model
# model.save(sc, "target/tmp/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
