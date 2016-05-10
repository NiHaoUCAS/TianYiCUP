#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import numpy as np
import time
V=10
DAY=7  #one week has 7 days
#sigle sample form [userID,webID,day]
train_X=[]
train_Y=[]
test_X=[]
test_Y=[]
final_X=[]

#training set :week 0,1,2->3 1,2,3->4 
def train_set_extract(dic):
	start_time=time.time()
	count=0
	zero_count=0
	no_zero_count=0
	for week in [5]:
		for elem in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):					
					if (dic[elem][week][v][day]==0):
						count+=1
					if(dic[elem][week][v][day]==0 and count%6!=0):
						continue
					if (dic[elem][week][v][day]==0):
						zero_count+=1
					else:
						no_zero_count+=1
					single_sample=[elem,day,v]

					single_sample.append(sum(dic[elem][week-1][v])) #last week all day
					single_sample.append(sum(dic[elem][week-2][v])) #last last week all day
					single_sample.append(sum(dic[elem][week-3][v])) #last last last week all day
					
					single_sample.append(dic[elem][week-1][v][0])# last week
					single_sample.append(dic[elem][week-1][v][1])
					single_sample.append(dic[elem][week-1][v][2])
					single_sample.append(dic[elem][week-1][v][3])
					single_sample.append(dic[elem][week-1][v][4])
					single_sample.append(dic[elem][week-1][v][5])
					single_sample.append(dic[elem][week-1][v][6])	
					
					single_sample.append(dic[elem][week-2][v][0])# last last week
					single_sample.append(dic[elem][week-2][v][1])
					single_sample.append(dic[elem][week-2][v][2])
					single_sample.append(dic[elem][week-2][v][3])
					single_sample.append(dic[elem][week-2][v][4])
					single_sample.append(dic[elem][week-2][v][5])
					single_sample.append(dic[elem][week-2][v][6])	

					single_sample.append(dic[elem][week-3][v][0])# last last last day
					single_sample.append(dic[elem][week-3][v][1])
					single_sample.append(dic[elem][week-3][v][2])
					single_sample.append(dic[elem][week-3][v][3])
					single_sample.append(dic[elem][week-3][v][4])
					single_sample.append(dic[elem][week-3][v][5])
					single_sample.append(dic[elem][week-3][v][6])			
						
					train_X.append(single_sample)
					train_Y.append(dic[elem][week][v][day])
	print 'zero_count=',zero_count
	print 'no_zero_count=',no_zero_count
	print 'train set extract successfully!!!'
	print '1:',1.0*zero_count/no_zero_count
	print 'train set extract took %fs!' % (time.time() - start_time)
	return train_X , train_Y

#test set :week 2,3,4->5 
def test_set_extract(dic):
	start_time=time.time()
	for week in [6]:
		for elem in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					single_sample=[elem,day,v]
					
					single_sample.append(sum(dic[elem][week-1][v])) #last week all day
					single_sample.append(sum(dic[elem][week-2][v])) #last last week all day
					single_sample.append(sum(dic[elem][week-3][v])) #last last last week all day
					
					single_sample.append(dic[elem][week-1][v][0])# last week
					single_sample.append(dic[elem][week-1][v][1])
					single_sample.append(dic[elem][week-1][v][2])
					single_sample.append(dic[elem][week-1][v][3])
					single_sample.append(dic[elem][week-1][v][4])
					single_sample.append(dic[elem][week-1][v][5])
					single_sample.append(dic[elem][week-1][v][6])	
					
					single_sample.append(dic[elem][week-2][v][0])# last last week
					single_sample.append(dic[elem][week-2][v][1])
					single_sample.append(dic[elem][week-2][v][2])
					single_sample.append(dic[elem][week-2][v][3])
					single_sample.append(dic[elem][week-2][v][4])
					single_sample.append(dic[elem][week-2][v][5])
					single_sample.append(dic[elem][week-2][v][6])	
					
					single_sample.append(dic[elem][week-3][v][0])# last last last day
					single_sample.append(dic[elem][week-3][v][1])
					single_sample.append(dic[elem][week-3][v][2])
					single_sample.append(dic[elem][week-3][v][3])
					single_sample.append(dic[elem][week-3][v][4])
					single_sample.append(dic[elem][week-3][v][5])
					single_sample.append(dic[elem][week-3][v][6])
					
					test_X.append(single_sample)
					test_Y.append(dic[elem][week][v][day])
	print 'test set extract successfully!!!'
	print 'test set extract took %fs!' % (time.time() - start_time)
	return test_X , test_Y
	
def final_set_extract(dic):
	start_time=time.time()
	for week in [7]:
		for elem in dic:  #for each user
			for day in range(DAY): #for each web
				for v in range(V):
					single_sample=[elem,day,v]
					single_sample.append(sum(dic[elem][week-1][v])) #last week all day
					single_sample.append(sum(dic[elem][week-2][v])) #last last week all day
					single_sample.append(sum(dic[elem][week-3][v])) #last last last week all day
					
					single_sample.append(dic[elem][week-1][v][0])# last week
					single_sample.append(dic[elem][week-1][v][1])
					single_sample.append(dic[elem][week-1][v][2])
					single_sample.append(dic[elem][week-1][v][3])
					single_sample.append(dic[elem][week-1][v][4])
					single_sample.append(dic[elem][week-1][v][5])
					single_sample.append(dic[elem][week-1][v][6])	
					
					single_sample.append(dic[elem][week-2][v][0])# last last week
					single_sample.append(dic[elem][week-2][v][1])
					single_sample.append(dic[elem][week-2][v][2])
					single_sample.append(dic[elem][week-2][v][3])
					single_sample.append(dic[elem][week-2][v][4])
					single_sample.append(dic[elem][week-2][v][5])
					single_sample.append(dic[elem][week-2][v][6])	
	
					single_sample.append(dic[elem][week-3][v][0])# last last last day
					single_sample.append(dic[elem][week-3][v][1])
					single_sample.append(dic[elem][week-3][v][2])
					single_sample.append(dic[elem][week-3][v][3])
					single_sample.append(dic[elem][week-3][v][4])
					single_sample.append(dic[elem][week-3][v][5])
					single_sample.append(dic[elem][week-3][v][6])
					
					final_X.append(single_sample)
	
	print 'final set extract successfully!!!'
	print 'final set extract took %fs!' % (time.time() - start_time)
	return final_X 
'''
    
    seg=' '
    a=['1','2','nihaoma']
    train_X_Y=open('train_X_Y','w')
    train_X_Y.write(seg.join(a)+'\n')
    train_X_Y.write(seg.join(a)+'\n')
    train_X_Y.close()

if __name__ == '__main__':
    feature_extract()
'''
