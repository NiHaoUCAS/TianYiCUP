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
		
		#train_X=preprocessing.scale(train_X*1.0)		
		test_X=np.load('data/test_X_55.npy')
		test_Y=np.load('data/test_Y.npy')
		
		#test_X=preprocessing.scale(test_X*1.0)
		#train_X=np.concatenate((train_X,train_X[:,17:47].copy()),axis=1)
		#test_X=np.concatenate((test_X,test_X[:,17:47].copy()),axis=1)
		print train_X.shape
		print test_X.shape
		train_X=train_X*1.0
		test_X=test_X*1.0
		#train_X[:,55:]=preprocessing.scale(train_X[:,55:]*1.0)
		#test_X[:,55:]=preprocessing.scale(test_X[:,55:]*1.0)
		'''
		for i in range(2*num):
			for j in range(6):
				if(np.sum(train_X[i,17+5*j:22+5*j])>0):
					train_X[i,17+5*j:22+5*j]=train_X[i,17+5*j:22+5*j]/np.sum(train_X[i,17+5*j:22+5*j])
					print '(%d/%d,%d/6)\r'%(i,2*num,j),
					sys.stdout.flush()
		for i in range(num):
			for j in range(6):
				if(np.sum(test_X[i,17+5*j:22+5*j])>0):
					test_X[i,17+5*j:22+5*j]=test_X[i,17+5*j:22+5*j]/np.sum(test_X[i,17+5*j:22+5*j])
		'''
		#train_X[:,17:47]=preprocessing.scale(train_X[:,17:47])	
		#test_X[:,17:47]=preprocessing.scale(test_X[:,17:47])
		scaler = preprocessing.StandardScaler().fit(train_X[:,17:47])
		train_X[:,17:47]=scaler.transform(train_X[:,17:47])
		test_X[:,17:47]=scaler.transform(test_X[:,17:47])
	if RELEASE==1:
		train_X=np.load('data/train_X_online_55.npy')
		train_X=np.concatenate((train_X,train_X[:,17:47].copy()),axis=1)
		train_Y=np.load('data/train_Y_online.npy')
		#train_X=preprocessing.scale(train_X*1.0)
		train_X=train_X*1.0
		for i in range(2*num):
			for j in range(6):
				if(np.sum(train_X[i,17+5*j:22+5*j])>0):
					train_X[i,17+5*j:22+5*j]=train_X[i,17+5*j:22+5*j]/np.sum(train_X[i,17+5*j:22+5*j])
					print '(%d/%d,%d/6)\r'%(i,2*num,j),
					sys.stdout.flush()
	if RELEASE==1:
		final_X= np.load('data/predict_X_55.npy')
		#final_X=preprocessing.scale(final_X*1.0)
		final_X=np.concatenate((final_X,final_X[:,17:47].copy()),axis=1)
		final_X=final_X*1.0
		for i in range(num):
			for j in range(6):
				if(np.sum(final_X[i,17+5*j:22+5*j])>0):
					final_X[i,17+5*j:22+5*j]=final_X[i,17+5*j:22+5*j]/np.sum(final_X[i,17+5*j:22+5*j])
	train_Y_trans=train_Y.copy()
	train_Y_trans=train_Y_trans.reshape((num*2,70))
	for i in range(2*num):
		if np.sum(train_Y_trans[i,:]) != 0:
			train_Y_trans[i,:]=train_Y_trans[i,:]*1.0/np.sum(train_Y_trans[i,:])
	train_Y_trans=train_Y_trans.reshape(num*70*2)
	print 'feature dim=%d' % train_X.shape[1]
	print 'load data took %fs!' % (time.time() - start_time)

	for n_estimators in [200]:
		for max_depth in [7]:
			for learning_rate in [1]:		
				model = other.random_forest_regressor(train_X, train_Y_trans*10000, n_estimators,  max_depth, random_state=0,n_jobs=20 )
				#print model.feature_importances_ 
				start_time=time.time()
				
				if DEBUG==1 or RELEASE==1:
					print 'predict begining...'
					test_Y_predict=model.predict(train_X)
					#test_Y_predict=other.modify(num,5,test_Y_predict)
					for i in range(len(test_Y_predict)):
						if test_Y_predict[i]<0.5:
							test_Y_predict[i]=0
					print 'save predict file'
					other.write_result_file(map,train_X,test_Y_predict)				
					other.make_reference(map,train_X,train_Y)
					print 'train set**********************'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					#get result
					test.get_result()
				
				if RELEASE==1:
					print 'predict begining...'
					final_Y_predict=model.predict(final_X)
					#final_Y_predict=other.modify(num,7,final_Y_predict)
					for i in range(len(final_Y_predict)):
						if final_Y_predict[i]<0.5:
							final_Y_predict[i]=0
					#zero_scale=np.load('data/delete_predict.npy')
					#final_Y_predict=final_Y_predict.reshape((num,70))
					#for i in range(num):
					#	if zero_scale[i]==0:
					#		final_Y_predict[i,:]=0
					#final_Y_predict=final_Y_predict.reshape(num*70)	
					#final_Y_predict=other.zeros_scale(final_Y_predict)
					print 'save predict file'
					other.write_result_file(map,final_X,final_Y_predict)
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
					other.write_result_file(map,test_X,test_Y_predict)
					other.make_reference(map,test_X,test_Y)
					print 'test set*******'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					test.get_result()
				
				print '*******************************\n'
	
				

	
	
	
