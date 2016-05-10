#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
from sklearn import metrics
import cPickle as pickle
from sklearn import preprocessing
import math
import xgboost as xgb
import feature_extract_and_make_sample as sample
import other as other
import test as test


DEBUG=1
RELEASE=0

	
	

if __name__ == '__main__':
	if DEBUG==1 and RELEASE==0:
		print 'DEBUG mode'
	if DEBUG==0 and RELEASE==1:
		print 'RELEASE mode'
	map_file=open('data/map.pkl','rb')
	map=pickle.load(map_file)
	map_file.close()
	num=len(map)
	print 'num=%d' % num
	start_time=time.time()	
	if DEBUG==1:
		train_X=np.load('data/train_X_55.npy')
		train_Y=np.load('data/train_Y.npy')
		train_behavior_X=np.load('data/train_behavior_V_X.npy')
		#train_X_day=np.load('data/train_X_day.npy')
		#train_X=preprocessing.scale(train_X*1.0)
		test_X=np.load('data/test_X_55.npy')
		test_Y=np.load('data/test_Y.npy')
		test_behavior_X=np.load('data/test_behavior_V_X.npy')
		#test_X_day=np.load('data/test_X_day.npy')
		#test_X_day=np.load('data/test_X_day.npy')
		#test_X=preprocessing.scale(test_X*1.0)
		train_X=np.concatenate((train_X,train_behavior_X),axis=1)
		print 'delete train_behavior_V_X'
		del train_behavior_X
		test_X=np.concatenate((test_X,test_behavior_X),axis=1)
		print 'delete test_behavior_V_X'
		del test_behavior_X
		print train_X.shape
		print test_X.shape
		train_X=train_X*1.0
		test_X=test_X*1.0
		'''
		L=train_X.shape[0]
		for j in range(6):
			SUM=train_X[:,17+5*j:22+5*j].sum(axis=1)+0.001
			SUM=SUM.reshape((L,1))
			train_X[:,17+5*j:22+5*j]=train_X[:,17+5*j:22+5*j]/SUM
		for j in range(6):
			SUM=train_X[:,55+5*j:60+5*j].sum(axis=1)+0.001
			SUM=SUM.reshape((L,1))
			train_X[:,55+5*j:60+5*j]=train_X[:,55+5*j:60+5*j]/SUM
		'''	
		#for i in range(train_X.shape[0]):
		#	for j in range(6):
		#		if(np.sum(train_X[i,17+5*j:22+5*j])>0):
		#			train_X[i,17+5*j:22+5*j]=train_X[i,17+5*j:22+5*j]/np.sum(train_X[i,17+5*j:22+5*j])
			#	if(np.sum(train_X[i,55+5*j:60+5*j])>0):
			#		train_X[i,55+5*j:60+5*j]=train_X[i,55+5*j:60+5*j]/np.sum(train_X[i,55+5*j:60+5*j])
		#		print '(%d/%d,%d/6)\r'%(i,train_X.shape[0],j),
		#		sys.stdout.flush()
		#print '\n'
		'''
		L=test_X.shape[0]
		for j in range(6):
			SUM=test_X[:,17+5*j:22+5*j].sum(axis=1)+0.001
			SUM=SUM.reshape((L,1))
			test_X[:,17+5*j:22+5*j]=test_X[:,17+5*j:22+5*j]/SUM
		for j in range(6):
			SUM=test_X[:,55+5*j:60+5*j].sum(axis=1)+0.001
			SUM=SUM.reshape((L,1))
			test_X[:,55+5*j:60+5*j]=test_X[:,55+5*j:60+5*j]/SUM
		print 'done!'
		'''
		#for i in range(test_X.shape[0]):
		#	for j in range(6):
		#		if(np.sum(test_X[i,17+5*j:22+5*j])>0):
		#			test_X[i,17+5*j:22+5*j]=test_X[i,17+5*j:22+5*j]/np.sum(test_X[i,17+5*j:22+5*j])
			#	if(np.sum(test_X[i,55+5*j:60+5*j])>0):
			#		test_X[i,55+5*j:60+5*j]=test_X[i,55+5*j:60+5*j]/np.sum(test_X[i,55+5*j:60+5*j])
		scaler = preprocessing.StandardScaler().fit(train_X[:,17:47])
		train_X[:,17:47]=scaler.transform(train_X[:,17:47])
		test_X[:,17:47]=scaler.transform(test_X[:,17:47])
		scaler = preprocessing.StandardScaler().fit(train_X[:,55:])
		train_X[:,55:]=scaler.transform(train_X[:,55:])
		test_X[:,55:]=scaler.transform(test_X[:,55:])
		#scaler = preprocessing.StandardScaler().fit(train_X[:,55:85])
		#train_X[:,55:85]=scaler.transform(train_X[:,55:85])
		#test_X[:,55:85]=scaler.transform(test_X[:,55:85])
	if RELEASE==1:
		train_X=np.load('data/train_X_online_55.npy')
		train_Y=np.load('data/train_Y_online.npy')
		#train_X=preprocessing.scale(train_X*1.0)
		train_X=train_X*1.0
		for i in range(2*num):
			for j in range(6):
				if(np.sum(train_X[i,17+5*j:22+5*j])>0):
					train_X[i,17+5*j:22+5*j]=train_X[i,17+5*j:22+5*j]/np.sum(train_X[i,17+5*j:22+5*j])
		scaler = preprocessing.StandardScaler().fit(train_X[:,17:47])
		train_X[:,17:47]=scaler.transform(train_X[:,17:47])
	if RELEASE==1:
		final_X= np.load('data/predict_X_55.npy')
		#final_X=preprocessing.scale(final_X*1.0)
		final_X=final_X*1.0
		for i in range(num):
			for j in range(6):
				if(np.sum(final_X[i,17+5*j:22+5*j])>0):
					final_X[i,17+5*j:22+5*j]=final_X[i,17+5*j:22+5*j]/np.sum(final_X[i,17+5*j:22+5*j])
		final_X[:,17:47]=scaler.transform(final_X[:,17:47])
	train_Y_trans=train_Y.copy()
	train_Y_trans=train_Y_trans.reshape((len(train_Y)/70,70))
	train_Y_trans=train_Y_trans/(train_Y_trans.sum(axis=1)+0.0001).reshape((2*num,1))
	train_Y_trans=train_Y_trans.reshape(2*num*70)

	print 'feature dim=%d' % train_X.shape[1]
	print 'load data took %fs!' % (time.time() - start_time)
	print 'make matrix begin'
	start_time=time.time()
	dtrain = xgb.DMatrix( train_X, label=train_Y_trans*10000)
	del train_X
	dtest = xgb.DMatrix(test_X,label=test_Y)
	del test_X
	print 'make matrix end'
	print 'make matirx took %fs!' % (time.time() - start_time)
	for n_estimators in [200]:
		for max_depth in [5,6,7,8]:
			for learning_rate in [0.01]:
				lambda_=100
				param = {'bst:max_depth':max_depth, 'bst:eta':learning_rate, 'silent':1, 'objective':'reg:linear', 'subsample':0.7,'colsample_bytree':0.3,'lambda':lambda_}
				#param['nthread'] = 10
				plst = param.items()
				#plst += [('eval_metric', 'rmse')]
				num_round=n_estimators
				start_time=time.time()
				model=xgb.train(plst,dtrain,num_round)
				print 'train model took %fs!' % (time.time() - start_time)
				start_time=time.time()
				if DEBUG==1 or RELEASE==1:
					print 'predict begining...'
					test_Y_predict=model.predict(dtrain)
					#test_Y_predict=other.modify(num,5,test_Y_predict)
					for i in range(len(test_Y_predict)):
						if test_Y_predict[i]<0.5:
							test_Y_predict[i]=0
					print 'save predict file'
					other.write_result_file(map,test_Y_predict)				
					other.make_reference(map,train_Y)
					print 'train set**********************'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth,'lambda',lambda_
					#get result
					test.get_result()
					
				if RELEASE==1:
					print 'predict begining...'
					final_Y_predict=model.predict(xgb.DMatrix(final_X))
					#final_Y_predict=other.modify(num,7,final_Y_predict)
					for i in range(len(final_Y_predict)):
						if final_Y_predict[i]<0.5:
							final_Y_predict[i]=0
					print 'save predict file'
					other.write_result_file(map,final_X,final_Y_predict)
					print 'save predict file finished!'
					exit()
				
				print '*******'
				#------------------------------------------------------------------------------------------------------#
				
				if DEBUG==1:
					start_time=time.time()
					test_Y_predict=model.predict(dtest)
					#test_Y_predict=other.modify(num,6,test_Y_predict)
					for i in range(len(test_Y_predict)):
						if test_Y_predict[i]<0.5:
							test_Y_predict[i]=0
					print 'predict took %fs!' % (time.time() - start_time)				
					print 'save predict file'				
					other.write_result_file(map,test_Y_predict)
					other.make_reference(map,test_Y)
					print 'test set*******'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					test.get_result()
				
				print '*******************************\n'
	
					
	
	
