#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_squared_error

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y, n_estimators, learning_rate, max_depth, random_state, loss ):
	start_time=time.time()
	from sklearn.ensemble import GradientBoostingRegressor
	model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state, loss=loss).fit(train_x, train_y)
	print 'training took %fs!' % (time.time() - start_time)
	return model
	
def write_result_file(map,test_X,test_Y_predict):
	result=open('result.txt','w')
	offset=0
	seg=','
	user_num=int(len(test_Y_predict)/70)
	for i in range(user_num):
		result_part=[int(test_Y_predict[i]+0.5) for i in range(offset,offset+70)]
		if(sum(result_part) != 0):
			result.write(map[test_X[offset][0]]+'\t')
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


def make_reference(map,dic_part,week_th):
	WEEK=7
	DAY=7
	V=10
	result=open('reference.txt','w')
	for elem in dic_part:
		if(np.sum(dic_part[elem][week_th])!=0):
			result.write(map[elem]+'\t')
			for i in range(DAY):
				for j in range(V):
					if((i!=(DAY-1)) or (j!=(V-1))):
						result.write(str(int(dic_part[elem][week_th][j][i]))+',')
					else:
						result.write(str(int(dic_part[elem][week_th][j][i]))+'\n')
				
				
	result.close()