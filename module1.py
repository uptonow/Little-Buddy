#from surprise import SVD
#from surprise import Dataset
#from surprise import accuracy
#from surprise.model_selection import train_test_split

## Load the movielens-100k dataset (download it if needed),
#data = Dataset.load_builtin('ml-100k')

## sample random trainset and testset
## test set is made of 25% of the ratings.
#trainset, testset = train_test_split(data, test_size=.25)

## We'll use the famous SVD algorithm.
#algo = SVD()

## Train the algorithm on the trainset, and predict ratings for the testset
#predictions = algo.fit(trainset).test(testset)

## Then compute RMSE
#accuracy.rmse(predictions)

#from surprise import KNNBasic
#from surprise import Dataset

#data = Dataset.load_builtin('ml-100k')
#trainset = data.build_full_trainset()
#algo = KNNBasic()
#algo.fit(trainset)
#uid = str(196)
#iid = str(302)
#pred = algo.predict(uid,iid,r_ui=4,verbose=True)

#import pandas as pd
#from surprise import NormalPredictor
#from surprise import Dataset
#from surprise import Reader
#from surprise.model_selection import cross_validate

#ratings_dict = {'itemID':[1,1,1,2,2],
#                'userID':[9,32,2,45,'user_foo'],
#                'rating':[3,2,4,3,1]}
#df = pd.DataFrame(ratings_dict)
#print(df)
#reader = Reader(rating_scale=(1,5))
#data = Dataset.load_from_df(df[['userID','itemID','rating']],reader)
#cross_validate(NormalPredictor(),data,cv=2)

#from surprise import SVD
#from surprise import Dataset
#from surprise import accuracy
#from surprise.model_selection import RepeatedKFold

#data = Dataset.load_builtin('ml-100k')
#kf = RepeatedKFold(n_splits=3,n_repeats=20)
#algo = SVD()
#for trainset, testset in kf.split(data):
#    algo.fit(trainset)
#    predictions = algo.test(testset)
#    accuracy.rmse(predictions, verbose=True)

#from surprise import SVD
#from surprise import Dataset
#from surprise.model_selection import GridSearchCV
#import pandas as pd

#data = Dataset.load_builtin('ml-100k')
#param_grid = {'n_epochs': [5,10], 'lr_all': [0.002,0.005],
#             'reg_all':[0.4,0.6]}
#gs = GridSearchCV(SVD,param_grid,measures=['rmse','mae'],cv=3)
#gs.fit(data)
##print(gs.best_score['rmse'])
##print(gs.best_params['rmse'])
#algo = gs.best_estimator['rmse']
#algo.fit(data.build_full_trainset())
#result_df = pd.DataFrame.from_dict(gs.cv_results)
#print(result_df)

from surprise import AlgoBase, accuracy
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import numpy as np
from surprise import PredictionImpossible

class MyOwnAlgorithm(AlgoBase):
    def __init__(self,sim_options={}, bsl_options={},verbose = False):
        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options,verbose = False)

    def fit(self,trainset):
        AlgoBase.fit(self,trainset)
        #self.the_mean = np.mean([r for (_,_,r) in self.trainset.all_ratings()])
        self.verbose = False
        self.bu,self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        return self

    def estimate(self, u ,i):
        #sum_means = self.trainset.global_mean
        #div = 1
        #if self.trainset.knows_user(u):
        #    sum_means += np.mean([r for (_,r) in self.trainset.ur[u]])
        #    div += 1
        #if self.trainset.knows_item(i):
        #    sum_means += np.mean([r for (_,r) in self.trainset.ir[u]])
        #    div += 1
        #return sum_means / div
        
        #return self.the_mean
        div = 0
        est = 0
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown .')
        neighbors = [(v,self.sim[u,v],r) for (v,r) in self.trainset.ir[i]]
        neighbors = sorted(neighbors,key=lambda x: x[1], reverse=True)
        print('The  3 nearest neighbors of user', str(u), 'are:')
        for v, sim_uv,r in neighbors[:3]:
            print('user{0:} with sim {1:1.2f}'.format(v,sim_uv))
            est += r
            div +=1
        est = est/div
        
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]
        return est


if __name__ == '__main__':
    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=.25)
    bsl_options = {'method':'sgd',
                   'learning_rate': .0005,}
    sim_options = {'name':'pearson_baseline',
                   'shrinkage':0}
    algo = MyOwnAlgorithm(sim_options = sim_options,bsl_options = bsl_options)
    predictions = algo.fit(trainset).test(testset)
    print(predictions)
    accuracy.rmse(predictions, verbose = True)
              