#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
import cPickle as pickle
from sklearn import preprocessing
import math
import feature_extract_and_make_sample as sample
import other as other
import test as test


DEBUG=0
RELEASE=1

	
	

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
		#train_0_6_X=np.load('data/train_0-6_X.npy')
		#train_0_12_X=np.load('data/train_0-12_X.npy')
		#train_0_18_X=np.load('data/train_0-18_X.npy')
		#train_X=np.concatenate((train_X,train_0_6_X,train_0_12_X,train_0_18_X),axis=1)
		#del train_0_6_X , train_0_12_X , train_0_18_X
		#train_X=preprocessing.scale(train_X*1.0)		
		test_X=np.load('data/test_X_55.npy')
		test_Y=np.load('data/test_Y.npy')
		#test_0_6_X=np.load('data/test_0-6_X.npy')
		#test_0_12_X=np.load('data/test_0-12_X.npy')
		#test_0_18_X=np.load('data/test_0-18_X.npy')
		#test_X=np.concatenate((test_X,test_0_6_X,test_0_12_X,test_0_18_X),axis=1)
		#del test_0_6_X , test_0_12_X , test_0_18_X
		
		#test_X=preprocessing.scale(test_X*1.0)
		
		print train_X.shape
		print test_X.shape
		#train_X=train_X*1.0
		#test_X=test_X*1.0
		L=train_X.shape[0]
		train_X[:L/2,17:22]=preprocessing.scale(train_X[:L/2,17:22]*1.0)
		train_X[L/2:,17:22]=preprocessing.scale(train_X[L/2:,17:22]*1.0)
		test_X[:,17:22]=preprocessing.scale(test_X[:,17:22]*1.0)

		train_X[:L/2,27:32]=preprocessing.scale(train_X[:L/2,27:32]*1.0)
		train_X[L/2:,27:32]=preprocessing.scale(train_X[L/2:,27:32]*1.0)
		test_X[:,27:32]=preprocessing.scale(test_X[:,27:32]*1.0)
		
		train_X[:L/2,37:42]=preprocessing.scale(train_X[:L/2,37:42]*1.0)
		train_X[L/2:,37:42]=preprocessing.scale(train_X[L/2:,37:42]*1.0)
		test_X[:,37:42]=preprocessing.scale(test_X[:,37:42]*1.0)
		
		'''
		L=train_X.shape[0]
		scaler = preprocessing.StandardScaler().fit(train_X[L/2:,17:22])
		train_X[:,17:22]=scaler.transform(train_X[:,17:22])
		test_X[:,17:22]=scaler.transform(test_X[:,17:22])
	
		scaler = preprocessing.StandardScaler().fit(train_X[L/2:,27:32])
		train_X[:,27:32]=scaler.transform(train_X[:,27:32])
		test_X[:,27:32]=scaler.transform(test_X[:,27:32])
	
		scaler = preprocessing.StandardScaler().fit(train_X[L/2:,37:42])
		train_X[:,37:42]=scaler.transform(train_X[:,37:42])
		test_X[:,37:42]=scaler.transform(test_X[:,37:42])
		#scaler = preprocessing.StandardScaler().fit(train_X[:,55:])
		#train_X[:,55:]=scaler.transform(train_X[:,55:])
		#test_X[:,55:]=scaler.transform(test_X[:,55:])
		'''
	if RELEASE==1:
		train_X=np.load('data/train_X_online_55.npy')
		train_Y=np.load('data/train_Y_online.npy')
		final_X= np.load('data/predict_X_55.npy')
		
	if RELEASE==1:
		'''
		L=train_X.shape[0]
		train_X[:L/2,17:22]=preprocessing.scale(train_X[:L/2,17:22]*1.0)
		train_X[L/2:,17:22]=preprocessing.scale(train_X[L/2:,17:22]*1.0)
		final_X[:,17:22]=preprocessing.scale(final_X[:,17:22]*1.0)

		train_X[:L/2,27:32]=preprocessing.scale(train_X[:L/2,27:32]*1.0)
		train_X[L/2:,27:32]=preprocessing.scale(train_X[L/2:,27:32]*1.0)
		final_X[:,27:32]=preprocessing.scale(final_X[:,27:32]*1.0)
		
		train_X[:L/2,37:42]=preprocessing.scale(train_X[:L/2,37:42]*1.0)
		train_X[L/2:,37:42]=preprocessing.scale(train_X[L/2:,37:42]*1.0)
		final_X[:,37:42]=preprocessing.scale(final_X[:,37:42]*1.0)
		'''
	
	train_Y_trans=train_Y.copy()
	train_Y_trans=train_Y_trans.reshape((num*2,70))
	for i in range(2*num):
		if np.sum(train_Y_trans[i,:]) != 0:
			train_Y_trans[i,:]=train_Y_trans[i,:]*1.0/np.sum(train_Y_trans[i,:])
	train_Y_trans=train_Y_trans.reshape(num*70*2)
	print 'feature dim=%d' % train_X.shape[1]
	print 'load data took %fs!' % (time.time() - start_time)

	for n_estimators in [700]:
		for max_depth in [7]:
			for learning_rate in [1]:		
				model = other.random_forest_regressor(train_X, train_Y_trans*10000, n_estimators,  max_depth, random_state=0,n_jobs=-1 )
				#print model.feature_importances_ 
				start_time=time.time()
				
				if DEBUG==1 or RELEASE==1:
					print 'predict begining...'
					test_Y_predict=model.predict(train_X)
					print model.feature_importances_
					#test_Y_predict=other.modify(num,5,test_Y_predict)
					for i in range(len(test_Y_predict)):
						if test_Y_predict[i]<0.5:
							test_Y_predict[i]=0
					print 'save predict file'
					other.write_result_file(map,test_Y_predict)				
					other.make_reference(map,train_Y)
					print 'train set**********************'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					#get result
					test.get_result()
				
				if RELEASE==1:
					print 'predict begining...'
					final_Y_predict=model.predict(final_X)
					for i in range(len(final_Y_predict)):
						if final_Y_predict[i]<0.5:
							final_Y_predict[i]=0
					print 'save predict file'
					other.write_result_file(map,final_Y_predict)
					print 'save predict file finished!'
					exit()
				
				print '*******'
				
				#------------------------------------------------------------------------------------------------------#
				
				if DEBUG==1:
					start_time=time.time()
					test_Y_predict=model.predict(test_X)
					for i in range(len(test_Y_predict)):
						if test_Y_predict[i]<0.5:
							test_Y_predict[i]=0
					print 'predict took %fs!' % (time.time() - start_time)				
					print 'save predict file'				
					#test_Y_predict=other.zeros_scale(test_Y_predict)
					#zero_scale=np.load('data/delete_test.npy')
					#test_Y_predict=test_Y_predict.reshape((num,70))
					#for i in range(num):
					#	if zero_scale[i]==0:
					#		test_Y_predict[i,:]=0
					#test_Y_predict=test_Y_predict.reshape(num*70)	
					other.write_result_file(map,test_Y_predict)
					other.make_reference(map,test_Y)
					print 'test set*******'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					test.get_result()
				
				print '*******************************\n'
	
				

	
	
	
