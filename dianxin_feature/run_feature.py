#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import os
import time
import multiprocessing
import numpy as np
from pandas import Series,DataFrame
import cPickle as pickle
import feature_extract_and_make_sample as sample
#import feature_extract as sample
#import read_data as read_df



DEBUG=1
RELEASE=0

ADD=0
	

	
if __name__ == '__main__':
	
	map_file=open('data/map.pkl','rb')
	map=pickle.load(map_file)
	map_file.close()
	#prepare
	#299320
	num=len(map)
	print 'num=%d' % num
	df=DataFrame(np.load('data/df.npy'),columns=['user','v','week','day','times'])
	if ADD==0:
		p1 = multiprocessing.Process(target = sample.data_extract, args = ('train',num,df))
		p2 = multiprocessing.Process(target = sample.data_extract, args = ('test',num,df))
		#p3 = multiprocessing.Process(target = sample.data_extract, args = ('predict',num,df))
		#p4 = multiprocessing.Process(target = sample.data_extract, args = ('train_online',num,df))
	if ADD==1:
		p1 = multiprocessing.Process(target = sample.feature_add, args = ('train',num,df))
		p2 = multiprocessing.Process(target = sample.feature_add, args = ('test',num,df))
		p3 = multiprocessing.Process(target = sample.feature_add, args = ('predict',num,df))
		p4 = multiprocessing.Process(target = sample.feature_add, args = ('train_online',num,df))

	p1.start()
	p2.start()
	#p3.start()
	#p4.start()
	'''
	sample.data_extract('train',num,df)
	sample.data_extract('test',num,df)
	sample.data_extract('predict',num,df)
	'''


	
	
