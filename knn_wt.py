
import numpy as np
import pandas as pd
import sys

master_k= int(input('enter k value: '))



data_train = pd.read_csv('data/ratings_training_90.csv')
data_test = pd.read_csv('data/ratings_test_10.csv')
train_user_id = data_train.ix[:,0].values
train_movie_id = data_train.ix[:,1].values
col_from_test = 'euclidian_dist'

predict_val_list = []
zip_col = ' [(user_id, distance)] using k =' +str(master_k)
user_id_zip =[]
#for row_num_test in range(0,data_test.shape[0]):
weight_cal_list=[]
di_list=[]
k = master_k
rating_matrix=data_train.pivot(index='user_id',columns='movie_id', values='ratings').fillna(value=0)

for row_num_test in range(0,data_test.shape[0]):
    target_user_id = data_test.ix[row_num_test,0]
    target_movie_id = data_test.ix[row_num_test,1]
    people_watched_film =data_train.ix[data_train.movie_id == target_movie_id ].ix[:,0].values
    data_train_temp = data_train.ix[data_train.movie_id == target_movie_id ]
    subset_movie_ratings =  rating_matrix.ix[people_watched_film,:] # gives subset of people who watched the target film
    target_user_rating  = rating_matrix.ix[ target_user_id,:].values # all the ratings the target user watched
    ecud_dist_array = ((subset_movie_ratings.values - target_user_rating)**2).sum(axis =1) #sqare root take it

    data_train_temp[col_from_test] = pd.Series( ecud_dist_array , index=data_train_temp.index)

     ## weights part modification
    if data_train_temp.shape[0]< k and data_train_temp.shape[0]>1  :
        k = data_train_temp.shape[0]
        weight_temp = data_train_temp.sort([col_from_test]).head(k).values[:,4]
        weight_temp_max = weight_temp[k-1]
        weight_temp_min = weight_temp[0]
        weights_cal = (weight_temp_max - weight_temp)/ (weight_temp_max - weight_temp_min)
        di= data_train_temp.sort([col_from_test]).head(k).values[:,4] *weights_cal
        predict_with_weights= np.sum(di* data_train_temp.sort([col_from_test]).head(k).values[:,2])/np.sum(di)
        print('k : ', k, ' if_condi_num', 1,' ,test_row_num :', row_num_test ,', predict_val :'  ,predict_with_weights )

    if data_train_temp.shape[0] == 1:
        weights_cal = 1
        k=1
        di=1
        predict_with_weights = data_train_temp.iloc[0]['ratings']
        print('k : ', k, ' if_condi_num', 2,' ,test_row_num :', row_num_test ,', predict_val :'  ,predict_with_weights )


    if data_train_temp.shape[0] > k :
        weight_temp = data_train_temp.sort([col_from_test]).head(k).values[:,4]
        weight_temp_max = weight_temp[k-1]
        weight_temp_min = weight_temp[0]
        weights_cal = (weight_temp_max - weight_temp)/ (weight_temp_max - weight_temp_min)
        di= data_train_temp.sort([col_from_test]).head(k).values[:,4] *weights_cal
        predict_with_weights= np.sum(di* data_train_temp.sort([col_from_test]).head(k).values[:,2])/np.sum(di)
        print('k : ', k, ' if_condi_num', 3,' ,test_row_num :', row_num_test ,', predict_val :'  ,predict_with_weights )

    predict_val_list.append(round(predict_with_weights,1))  # append to list here
    #print(row_num_test, ' ) ', predict_with_weights)
    zip_val = list(zip(list(data_train_temp.sort([col_from_test]).head(k).values[:,0]),list(data_train_temp.sort([col_from_test]).head(k).values[:,4])))
    user_id_zip.append(zip_val)
    #weight_cal_list.append(weights_cal)
    #di_list.append(di)
    if k != master_k:
        k = master_k

predict_val = np.asarray(predict_val_list)
data_test['predicted_value'] = pd.Series(predict_val , index=data_test.index)
data_test[zip_col] = pd.Series(user_id_zip , index=data_test.index)
#data_test['DI'] = pd.Series(di_list , index=data_test.index)
#data_test['weights_calculated'] = pd.Series(weight_cal_list , index=data_test.index)


from sklearn.metrics import mean_squared_error
from math import sqrt


rms = sqrt(mean_squared_error(data_test.ix[:,2].values, data_test.ix[:,4].values))
target = open('result/output.txt','a+')

target.write('ratings_predictions_10_knn_wt_k'+str(master_k)+': RMSE :'+  str(rms)+'\n')

target.close()

data_test.to_csv('result/ratings_predictions_10_knn_wt_k'+str(master_k)+'.csv')
