#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
from pandas import DataFrame,Series
from sklearn.metrics import mean_squared_error

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y, n_estimators, learning_rate, max_depth, random_state, loss ):
	start_time=time.time()
	from sklearn.ensemble import GradientBoostingRegressor
	model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state, loss=loss).fit(train_x, train_y)
	print 'training took %fs!' % (time.time() - start_time)
	return model

def random_forest_regressor(train_x, train_y, n_estimators,  max_depth, random_state,n_jobs ):
	start_time=time.time()
	from sklearn.ensemble import RandomForestRegressor  #,
	model = RandomForestRegressor(n_estimators=n_estimators,max_features='log2',max_depth=max_depth,random_state=random_state ,n_jobs=n_jobs).fit(train_x, train_y)  # 
	print 'training took %fs!' % (time.time() - start_time)
	return model
	
def write_result_file(map,test_Y_predict):
	result=open('result.txt','w')
	offset=0
	seg=','
	user_num=int(len(test_Y_predict)/70)
	if user_num>300000:
		for user_id in range(user_num):
			result_part=[int(test_Y_predict[i]+0.5) for i in range(offset,offset+70)]
			if(sum(result_part) != 0):
				#result.write(map[test_X[offset][0]]+'\t')
				result.write(str(user_id)+'\t')
				result_part=[str(result_part[i]) for i in range(len(result_part))]
				result.write(seg.join(result_part)+'\n')
			offset+=70
	else :
		for user_id in range(user_num):
			result_part=[int(test_Y_predict[i]+0.5) for i in range(offset,offset+70)]
			if(sum(result_part) != 0):
				#result.write(map[test_X[offset][0]]+'\t')
				result.write(map[user_id]+'\t')
				result_part=[str(result_part[i]) for i in range(len(result_part))]
				result.write(seg.join(result_part)+'\n')
			offset+=70
	result.close()
	
def part_dic(dic,num):
	counter=0
	dic_part={}
	map={}
	for elem in dic:
		counter+=1
		if(counter>num):
			break
		dic_part[counter]=dic[elem]
		map[counter]=elem
	print counter-1 , 'elem in dic_part'
	return dic_part,map


def make_reference(map,test_Y):
	result=open('reference.txt','w')
	offset=0
	seg=','
	user_num=int(len(test_Y)/70)
	if user_num>300000:
		for user_id in range(user_num):
			result_part=[int(test_Y[i]+0.5) for i in range(offset,offset+70)]
			if(sum(result_part) != 0):
				#result.write(map[test_X[offset][0]]+'\t')
				result.write(str(user_id)+'\t')
				result_part=[str(result_part[i]) for i in range(len(result_part))]
				result.write(seg.join(result_part)+'\n')
			offset+=70
	else:
		for user_id in range(user_num):
			result_part=[int(test_Y[i]+0.5) for i in range(offset,offset+70)]
			if(sum(result_part) != 0):
				#result.write(map[test_X[offset][0]]+'\t')
				result.write(map[user_id]+'\t')
				result_part=[str(result_part[i]) for i in range(len(result_part))]
				result.write(seg.join(result_part)+'\n')
			offset+=70
	result.close()
	
def modify(num,week,test_Y_predict):
	DAY=7
	V=10
	count=0
	df=DataFrame(np.load('data/df.npy'),columns=['user','v','week','day','times'])
	Empty=df['times'].groupby([df['week']<week,df['user'],df['v']]).sum().values
	Empty=Empty[len(Empty)/2:]
	for user in range(num):
		for day in range(DAY):
			for v in range(V):
				if(Empty[user*V+v]==0):
					if(int(test_Y_predict[user*DAY*V+day*V+v]+0.5)>=1):
						count+=1
					test_Y_predict[user*DAY*V+day*V+v]=0
	print 'count=%d' % count
	return test_Y_predict
	
def zeros_scale(test_Y_predict):
	DAY=7
	V=10
	count=0
	df=DataFrame(np.load('data/df.npy'),columns=['user','v','week','day','times'])
	Empty=df['times'].groupby([df['week'],df['user']]).sum().values
	num=len(Empty)/7
	w0=Empty[0:num]
	w1=Empty[num:num*2]
	w2=Empty[num*2:num*3]
	w3=Empty[num*3:num*4]
	w4=Empty[num*4:num*5]
	w5=Empty[num*5:num*6]
	w6=Empty[num*6:num*7]
	scale0_3=(w5+w6)
	zero_index=np.array([i for i,a in enumerate(scale0_3) if a==0])
	print 'scale0_3=',len(zero_index)
	test_Y_predict=test_Y_predict.reshape((num,70))
	test_Y_predict[zero_index,:]=0
	test_Y_predict=test_Y_predict.reshape(num*70)
	return test_Y_predict
	

