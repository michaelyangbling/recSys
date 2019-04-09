import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from numpy import vectorize
import math

songs=pd.read_csv("msdchallenge/kaggle_songs.txt", sep=" ", header=None)
users=pd.read_csv("msdchallenge/kaggle_users.txt", header=None)

valHid = pd.read_csv("EvalDataYear1MSDWebsite/year1_valid_triplets_visible.txt", sep="\t", header=None)

valVis = pd.read_csv("EvalDataYear1MSDWebsite/year1_valid_triplets_hidden.txt", sep="\t", header=None)

testVis = pd.read_csv("EvalDataYear1MSDWebsite/year1_test_triplets_visible.txt", sep="\t", header=None)

testHid = pd.read_csv("EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt", sep="\t", header=None)




songs.columns=["song_id","song_index"]
users.columns=["user_id"]
valHid.columns = ["user_id","song_id","counts"]
valVis.columns = ["user_id","song_id","counts"]

testVis.columns=["user_id","song_id","counts"]
testHid.columns=["user_id","song_id","counts"]


print( list( valHid.user_id.unique() )==list( valVis.user_id.unique() ) ) #users same for val and test
print( list( testHid.user_id.unique() )==list( testVis.user_id.unique() ) )





testVis = testVis.head(10000) 
testHid = testHid.head(10000) #to accelerate following for loop 
valVis = valVis.head(2000)
valHid = valHid.head(2000)

valHid.counts = np.log1p(valHid.counts)
valVis.counts = np.log1p(valVis.counts)
testVis.counts = np.log1p(testVis.counts)
testHid.counts = np.log1p(testHid.counts)

train = pd.concat([testVis,testHid, valVis])

data = pd.concat([train, valHid, valVis])
# print( data.user_id.nunique() )
# print( data.song_id.nunique() )
# print( valHid.user_id.nunique() )
# print( valVis.user_id.nunique() )
# print( testHid.user_id.nunique() )
# print( testVis.user_id.nunique() )


subsongs = pd.Series( data.song_id.unique() )#.sample(frac=1/50, random_state=1)
subusers=pd.Series( data.user_id.unique() )#.sample(frac=0.2, random_state=1)
#to decrease matrix size and a random sampling, but this sampling can make matrix too sparse




matrix = pd.DataFrame( np.zeros( (subusers.__len__(), subsongs.__len__()) ) )
matrix.columns=list(subsongs)
matrix.index=list(subusers)
# setsongs=set(subsongs)
# setusers=set(subusers)


for i in range(train.__len__()): #iloc: position based, loc: index(or column name) based
    #print(i)
    r=train.iloc[i].loc['user_id']
    c=train.iloc[i].loc['song_id']
#     if r in setusers and c in setsongs:
    matrix.loc[r].loc[c]=train.iloc[i].loc['counts']
    #log scale conversion create extremly large predictions...should not take log


# Q = np.diag( np.sum(R, axis=0) )   #column sum as diag

# Q2 = vfunc(Q)

# res2 = R @ Q2 @ Rt @ R @ Q2

def getusermat(matrix, subsongs, subusers):
    vfunc = vectorize( lambda x: 0 if x==0 else 1/( math.sqrt(x) ) )
    R = matrix.values # userItem matrix R
    Rt = R.transpose()
    Q = np.diag( np.sum(R, axis=0) )   #col sum as diag

    Q2 = vfunc(Q)

    Rt = R.transpose()
    res = R @ Q2 @ Rt @ R @ Q2
    res = pd.DataFrame( res )
    res.columns=list(subsongs)
    res.index=list(subusers)
    return res

usermat = getusermat(matrix, subsongs, subusers)

pred=[]
for i in range( valHid.__len__() ):
    r=valHid.iloc[i].loc['user_id']
    c=valHid.iloc[i].loc['song_id']
    pred.append( usermat.loc[r].loc[c])
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = math.sqrt( mean_squared_error(pred, valHid.counts) ) #rms 50
print("testRms", rms)
valHid['pred'] = pred

def getmap(df):#get mean average precision
    d={uid:[[], []] for uid in df['user_id']} #[[count], [pred]]
    for i in range( df.__len__() ):
        series = df.iloc[i]
        userid=series.loc['user_id']
        d[userid][0].append( float(series.loc['counts']) )
        d[userid][1].append( float(series.loc['pred']) )
    isum=0
    count=0
#     print(d)
    for key in d:
        #print(d[key][0], d[key][1])
        if not math.isnan( spearmanr( d[key][0], d[key][1] )[0]):# zero varianve leads to nan
          isum+=spearmanr( d[key][0], d[key][1] )[0]
          count+=1
#         else:
#           print(d[key][0], d[key][1])
            
    return count, isum/count, d.__len__()  # test row counts, score, original test shape
print( "test row counts, test average spearman score, original test data counts", getmap(valHid) )


trainPred=[]
for i in range( train.__len__() ):
    r=train.iloc[i].loc['user_id']
    c=train.iloc[i].loc['song_id']
    trainPred.append( usermat.loc[r].loc[c])
from sklearn.metrics import mean_squared_error
from math import sqrt

trainRms = math.sqrt( mean_squared_error(trainPred, train.counts) ) #train rms 750?, rms not a good metric
print("trainRms", trainRms)
train['pred'] = trainPred 
print("train row counts, train average spearman score, original train data counts", getmap(train) )# at least training data's average spearman R does not suck like rms



#How data size influences performance

# testVis = testVis.head(10000) 
# testHid = testHid.head(10000) #to accelerate following for loop 
# valVis = valVis.head(2000)
# valHid = valHid.head(2000)
#item-item is better
# 
#tried log, but rmse of log converted back is same
# testRms 24.036403100845543
# trianRms: 1743.365554133911
# test result: (89, 0.041032111881166865, 163)
# train result: (825, 0.729534355568378, 890)



# testVis = testVis.head(3500) 
# testHid = testHid.head(3500) #to accelerate following for loop 
# valVis = valVis.head(1000)
# valHid = valHid.head(1000)
# testRms 11.717058579457346
# trianRms: 1335.4541556435092
# test result: (33, -0.031043499920132736, 85)
# train result: train row counts, train average spearman score, original train data counts (312, 0.7956095992402735, 338)




# testVis = testVis.head(1500) 
# testHid = testHid.head(1500) #to accelerate following for loop 
# valVis = valVis.head(500)
# valHid = valHid.head(500)
# testRms 13.841488793870228
# trainRms: 1570.809557704632
# test result: (11, -0.05573288843091613, 41)
# train result: (145, 0.8580814344493973, 152)






#log: computational slower, spearman Rank same, meaning log or not log give similar results
#so spearman maybe a better rank

# testVis = testVis.head(10000) 
# testHid = testHid.head(10000) #to accelerate following for loop 
# valVis = valVis.head(2000)
# valHid = valHid.head(2000)
#item-item is better
# 
#tried log, but rmse of log converted back is same
# testRms 1.2852925432055633
# trianRms: 46.804711543019735
# test result: (89, 0.04907224410554281, 163)
# train result: (825, 0.6089127292869666, 890)