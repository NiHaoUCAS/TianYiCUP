#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import numpy as np
import time
import threading
V=10
DAY=7  #one week has 7 days
#sigle sample form [userID,webID,day]


final_X=[]
#training set :week 0,1,2->3 1,2,3->4 
def data_set_extract(dic,data_type):
	data_X=[]
	data_Y=[]
	start_time=time.time()
	if data_type=='train':
		WEEK=[3,4]
	if data_type=='test':
		WEEK=[5]
	if data_type=='predict':
		WEEK=[7]
	#threads=[]
	#T=threading.Thread(target=feature1,args=(dic,WEEK,))
	#threads.append(T)	
	#T=threading.Thread(target=feature2,args=(dic,WEEK,))
	#threads.append(T)
	#T=threading.Thread(target=feature3,args=(dic,WEEK,))
	#threads.append(T)	
	#for T in threads:
	#	T.setDaemon(True)
	#	T.start()
	#for t in threads:
	#	T.join()
	if data_type!='predict':
		feature_num,feature=feature0(dic,WEEK)
		data_Y=feature
	
	feature_num,feature=feature1(dic,WEEK)
	if feature_num==1:
		data_X.append(feature)
	else:
		for i in range(feature_num):
			data_X.append(feature[i])
		
	feature_num,feature=feature2(dic,WEEK)
	if feature_num==1:
		data_X.append(feature)
	else:
		for i in range(feature_num):
			data_X.append(feature[i])
		
#	feature_num,feature=feature3(dic,WEEK)
#	if feature_num==1:
#		data_X.append(feature)
#	else:
#		for i in range(feature_num):
#			data_X.append(feature[i])
	data_X=(np.array(data_X)).transpose()
	print 'train set extract successfully!!!'
	print 'train set extract took %fs!' % (time.time() - start_time)
	if data_type=='predict':
		return data_X
	return data_X , data_Y
	
def feature0(dic,WEEK):
	feature_num=1
	feature_1=[]
	for week in WEEK:
		for user in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					feature_1.append(dic[user][week][v][day])
					
	return feature_num,(feature_1)
	
def feature1(dic,WEEK): #[user,day,v]
	feature_num=3
	feature_1=[]
	feature_2=[]
	feature_3=[]
	for week in WEEK:
		for user in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					feature_1.append(user)
					feature_2.append(day)
					feature_3.append(v)
					
	return feature_num,(feature_1,feature_2,feature_3)
	
def feature2(dic,WEEK):
	feature_num=6
	feature_1=[]
	feature_2=[]
	feature_3=[]
	feature_4=[]
	feature_5=[]
	feature_6=[]
	for week in WEEK:
		for user in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					feature_1.append(dic[user][week-1][v][day]) #last week same day
					feature_2.append(dic[user][week-2][v][day]) #last last week same day
					feature_3.append(dic[user][week-3][v][day]) #last last last week same day
					feature_4.append(sum(dic[user][week-1][v])) #last week all day
					feature_5.append(sum(dic[user][week-2][v])) #last last week all day
					feature_6.append(sum(dic[user][week-3][v])) #last last last week all day
	return feature_num,(feature_1,feature_2,feature_3,feature_4,feature_5,feature_6)
					
def feature3(dic,WEEK): #[user,day,v]
	feature_num=1
	feature_1=[]
	for week in WEEK:
		for user in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					if day != 0 :
						feature_1.append(dic[user][week][v][day-1])	# last day
					else :
						feature_1.append(dic[user][week-1][v][DAY-1])	# last day
	return feature_num,(feature_1)
	
#test set :week 2,3,4->5 

def test_set_extract(dic):
	test_X=[]
	test_Y=[]
	start_time=time.time()
	WEEK=[6]
	#threads=[]
	#T=threading.Thread(target=feature1,args=(dic,WEEK,))
	#threads.append(T)	
	#T=threading.Thread(target=feature2,args=(dic,WEEK,))
	#threads.append(T)
	#T=threading.Thread(target=feature3,args=(dic,WEEK,))
	#threads.append(T)	
	#for T in threads:
	#	T.setDaemon(True)
	#	T.start()
	#for t in threads:
	#	T.join()
	feature_num,feature=feature0(dic,WEEK)
	test_Y=feature
	
	feature_num,feature=feature1(dic,WEEK)
	if feature_num==1:
		test_X.append(feature)
	else:
		for i in range(feature_num):
			test_X.append(feature[i])
		
	feature_num,feature=feature2(dic,WEEK)
	if feature_num==1:
		test_X.append(feature)
	else:
		for i in range(feature_num):
			test_X.append(feature[i])
		
	#feature_num,feature=feature3(dic,WEEK)
	#if feature_num==1:
	#	test_X.append(feature)
	#else:
	#	for i in range(feature_num):
	#		test_X.append(feature[i])
	test_X=(np.array(test_X)).transpose()
	print 'test set extract successfully!!!'
	print 'test set extract took %fs!' % (time.time() - start_time)
	return test_X , test_Y
	
def final_set_extract(dic):
	final_X=[]
	start_time=time.time()
	WEEK=[7]
	#threads=[]
	#T=threading.Thread(target=feature1,args=(dic,WEEK,))
	#threads.append(T)	
	#T=threading.Thread(target=feature2,args=(dic,WEEK,))
	#threads.append(T)
	#T=threading.Thread(target=feature3,args=(dic,WEEK,))
	#threads.append(T)	
	#for T in threads:
	#	T.setDaemon(True)
	#	T.start()
	#for t in threads:
	#	T.join()

	
	feature_num,feature=feature1(dic,WEEK)
	if feature_num==1:
		final_X.append(feature)
	else:
		for i in range(feature_num):
			final_X.append(feature[i])
		
	feature_num,feature=feature2(dic,WEEK)
	if feature_num==1:
		final_X.append(feature)
	else:
		for i in range(feature_num):
			final_X.append(feature[i])
		
	#feature_num,feature=feature3(dic,WEEK)
	#if feature_num==1:
	#	final_X.append(feature)
	#else:
	#	for i in range(feature_num):
	#		final_X.append(feature[i])
	final_X=(np.array(final_X)).transpose()
	print 'test set extract successfully!!!'
	print 'test set extract took %fs!' % (time.time() - start_time)
	return final_X 
