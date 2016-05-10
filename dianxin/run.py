#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_squared_error
import math

import read_data
import feature_extract_and_make_sample as sample
#import feature_extract as sample
import other as other
import test as test

DEBUG=1
RELEASE=0


	

	
if __name__ == '__main__':
	
	#prepare
		
	dic=read_data.read_dat()
	'''
	dic_file=open('data/dic.pkl','wb')
	pickle.dump(dic,dic_file)
	dic_file.close()
	'''
	#dic_file=open('data/dic.pkl','rb')
	#dic=pickle.load(dic_file)
	#dic_file.close()
	if DEBUG==1:
		num=10000
	else:
		#num=1000
		num=len(dic)

	dic_part,map=other.part_dic(dic,num)
	
	#train_X , train_Y = sample.data_set_extract(dic_part,'train')
	train_X , train_Y = sample.train_set_extract(dic_part)
	if DEBUG==1:
		#test_X , test_Y = sample.data_set_extract(dic_part,'test')
		test_X , test_Y = sample.test_set_extract(dic_part)
	if RELEASE==1:
		#final_X= sample.data_set_extract(dic_part,'predict')
		final_X= sample.final_set_extract(dic_part)

	'''
	train_data_file=open('data/train.pkl','wb')
	test_data_file=open('data/test.pkl','wb')
	pickle.dump((train_X,train_Y),train_data_file)
	pickle.dump((test_X,test_Y),test_data_file)
	train_data_file.close()
	test_data_file.close()
	'''
	'''
	#read training and test data
	start_time=time.time()
	train_data_file=open('data/train.pkl','rb')
	train_X , train_Y = pickle.load(train_data_file)
	train_data_file.close()
	test_data_file=open('data/test.pkl','rb')
	test_X , test_Y = pickle.load(test_data_file)
	test_data_file.close()
	print 'read training and test data took %fs!' % (time.time() - start_time)
	'''

	#test_Y=[math.log(i+1) for i in test_Y]
	for n_estimators in [200]:
		for max_depth in range(5,6):
			for learning_rate in range(1,2):
	#training model
				#train_Y=[math.log(train_Y[i]+1) for i in range(len(train_Y))]
				#n_estimators=5
				learning_rate=learning_rate*0.1
				#max_depth=5
				random_state=0
				loss='ls'
				model=other.gradient_boosting_classifier(train_X, train_Y, n_estimators, learning_rate, max_depth, random_state, loss)
				
				
				#training set accurency
				'''
				test_Y_predict=model.predict(train_X[0:len(test_Y)])
				other.write_result_file(map,test_X,test_Y_predict)
				other.make_reference(map,dic_part,3)
				test.get_result()
				test_Y_predict=model.predict(train_X[len(test_Y):])
				other.write_result_file(map,test_X,test_Y_predict)
				other.make_reference(map,dic_part,4)
				test.get_result()
				'''
				
				
				start_time=time.time()
				if DEBUG==1:
					test_Y_predict=model.predict(test_X)
					#test_Y_predict=[math.exp(test_Y_predict[i])-1 for i in range(len(test_Y_predict))]
				if RELEASE==1:
					final_Y_predict=model.predict(final_X)
				print 'predict took %fs!' % (time.time() - start_time)
				
				#print mean_squared_error(test_Y, test_Y_predict)
				#test_Y_predict=[(math.exp(i)-1) for i in test_Y_predict]
				#save predict file
				print 'save predict file'
				if DEBUG==1:
					other.write_result_file(map,test_X,test_Y_predict)
				if RELEASE==1:
					other.write_result_file(map,final_X,final_Y_predict)
				
				#make reference 
				if DEBUG==1:
					other.make_reference(map,dic_part,6)
					print '*******'
					print 'n_estimators=',n_estimators,'learning_rate=',learning_rate,'max_depth=',max_depth
					#get result
					test.get_result()
				
				print '-------------------------\n'
				#test_Y_predict=model.predict(test_X)
				
	'''
	test_predict_file=open('data/test_Y_predict.pkl','wb')
	pickle.dump(test_Y_predict,test_predict_file)
	test_predict_file.close()
	'''
	
	
	
