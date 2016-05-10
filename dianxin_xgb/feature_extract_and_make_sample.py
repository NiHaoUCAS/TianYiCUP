#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import numpy as np
import time
from pandas import Series,DataFrame
from sklearn import preprocessing
import threading
V=10
DAY=7  #one week has 7 days
#sigle sample form [userID,webID,day]
train_X=[]
train_Y=[]
test_X=[]
test_Y=[]
final_X=[]

def data_extract(type,num,df):
	start_time=time.time()
	if type=='train':
		weeks=[4,5]
	if type=='test':
		weeks=[6]
	if type=='predict':
		weeks=[7]
	if type=='train_online':
		weeks=[5,6]	
	Data_X=[]
	Data_Y=[]
	for week in weeks:

		#basic feature : user_id day v
		Base_day=np.array(np.array(range(DAY)*V).reshape((V,DAY)).T.reshape(V*DAY).tolist()*num).reshape((num*V*DAY,1))
		Base_v=np.array(range(V)*num*DAY).reshape((num*V*DAY,1))
		enc = preprocessing.OneHotEncoder()
		enc.fit([[0],[1],[2],[3],[4],[5],[6]])
		Base_day=enc.transform(Base_day).toarray()
		enc.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
		Base_v=enc.transform(Base_v).toarray()
		X=np.concatenate((Base_day,Base_v),axis=1)
		print 'good'
		f0=feature0(df,week)
		f1=v_feature1(df,num,week)
		f2=v_feature2(df,num,week)
		f3=user_feature1(df,num,week)
		f4=user_feature2(df,num,week)
		f5=user_v_feature1(df,num,week)
		f6=user_v_feature1(df,num,week)
		f7=user_day_v_feature(df,week)
		f8=no_act(df,num,week)

		X=np.concatenate((X,f1,f2,f3,f4,f5,f6,f7,f8),axis=1)
		if(len(Data_X)==0):
			Data_X=X
		else:
			Data_X=np.concatenate((Data_X,X),axis=0)
		if type!='predict':
			Y=f0
			if(len(Data_Y)==0):
				Data_Y=Y
			else:
				Data_Y=np.concatenate((Data_Y,Y),axis=0)
	zero_count=len(np.array([i for i,a in enumerate(Data_Y) if a==0]))
	no_zero_count=len(Data_Y)-zero_count
	
	print '%s zero_count=%d' % (type,zero_count)
	print '%s no_zero_count=%d' % (type, no_zero_count)

	print '%s set extract took %fs!' % (type,(time.time() - start_time))
	Data_X=np.array(Data_X)
	if type=='predict':	
		np.save('data/predict_X.npy',Data_X)
		return Data_X
	if type== 'train':
		Data_Y=np.array(Data_Y)
		np.save('data/train_X.npy',Data_X)
		np.save('data/train_Y.npy',Data_Y)
	if type== 'test':
		Data_Y=np.array(Data_Y)
		np.save('data/test_X.npy',Data_X)
		np.save('data/test_Y.npy',Data_Y)
	if type== 'train_online':
		Data_Y=np.array(Data_Y)
		np.save('data/train_X_online.npy',Data_X)
		np.save('data/train_Y_online.npy',Data_Y)
def feature_add(type,num,df):
	start_time=time.time()
	if type=='train':
		weeks=[4,5]
	if type=='test':
		weeks=[6]
	if type=='predict':
		weeks=[7]
	if type=='train_online':
		weeks=[5,6]	
	Data_X=[]
	for week in weeks:
		#f1=user_feature2(df,num,week)
		f2=no_act_add(df,num,week)
		#X=np.concatenate((f1,f2),axis=1)
		X=f2
		if(len(Data_X)==0):
			Data_X=X
		else:
			Data_X=np.concatenate((Data_X,X),axis=0)

	print '%s set extract took %fs!' % (type,(time.time() - start_time))
	Data_X=np.array(Data_X)
	if type=='predict':	
		predict_X=np.load('data/predict_X.npy')
		Data_X=np.concatenate((predict_X,Data_X),axis=1)
		np.save('data/predict_X_55.npy',Data_X)
	if type== 'train':
		train_X=np.load('data/train_X.npy')
		Data_X=np.concatenate((train_X,Data_X),axis=1)
		np.save('data/train_X_55.npy',Data_X)
	if type== 'test':
		test_X=np.load('data/test_X.npy')
		Data_X=np.concatenate((test_X,Data_X),axis=1)
		np.save('data/test_X_55.npy',Data_X)
	if type== 'train_online':
		train_X_online=np.load('data/train_X_online.npy')
		Data_X=np.concatenate((train_X_online,Data_X),axis=1)
		np.save('data/train_X_online_55.npy',Data_X)
	print Data_X.shape

	
def downsample(df,X,Y,scale):
	#get no act index
	no_act_index=[]
	Empty=df['times'].groupby([df['user'],df['v']]).sum().values
	no_act_index=np.array([i for i,a in enumerate(Empty) if a==0])
	u=no_act_index/V
	basic=no_act_index-u*V
	no_act_index=basic+u*V*DAY
	no_act_index=np.concatenate((no_act_index,no_act_index+10,no_act_index+20,no_act_index+30,no_act_index+40,no_act_index+50,no_act_index+60),axis=0)
	no_act_index=np.concatenate((no_act_index,no_act_index+len(Y)/2),axis=0).tolist()   #2 week

	all_index=range(len(Y))
	keep_index=list(set(all_index)^set(no_act_index))
	X=np.array(X)
	Y=np.array(Y)
	X=X[keep_index,:]
	Y=Y[keep_index]
	zero_index=np.array([i for i,a in enumerate(Y) if a==0])
	zero_count=len(zero_index)
	no_zero_count=len(Y)-zero_count
	span=int(zero_count/scale/no_zero_count+0.5)
	no_zero_index=list(set(range(len(Y)))^set(zero_index))
	zero_index_choice=np.arange(len(zero_index)/span)*span
	zero_index_keep=zero_index[zero_index_choice]
	all_keep=list(no_zero_index)+list(zero_index_keep)
	X=X[all_keep,:]
	Y=Y[all_keep]
	zero_count=len([i for i,a in enumerate(Y) if a==0])
	no_zero_count=len(Y)-zero_count
	print 'zero_count=%d' % zero_count
	print 'no_zero_count=%d' %no_zero_count
	return X,Y
	
#Y
def feature0(df,week):    
	f0=df['times'].groupby([df['week']==week,df['user'],df['day'],df['v']]).sum()
	f0=f0.values[len(f0)/2:]
	return f0

def v_feature1(df,num,week):
	V_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['v']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['v']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df['times'].groupby([df['week']==week-1,df['day']==5,df['v']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df['times'].groupby([df['week']==week-1,df['day']==4,df['v']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df['times'].groupby([df['week']==week-1,df['v']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df['times'].groupby([df['week']==week-1 ,df['v']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df['times'].groupby([df['week']==week-2,df['v']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df['times'].groupby([df['week']==week-3 ,df['v']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df['times'].groupby([df['week']==week-4 ,df['v']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	
	f1=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	f1=np.array(f1.tolist()*num*DAY).reshape((num*V*DAY,5))
	return f1
	
def v_feature2(df,num,week):
	df_bool=df.copy()
	df_bool['times']=df['times']>0
	
	V_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['v']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['v']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==5,df_bool['v']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==4,df_bool['v']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['v']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df_bool['times'].groupby([df_bool['week']==week-1 ,df_bool['v']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df_bool['times'].groupby([df_bool['week']==week-2,df_bool['v']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df_bool['times'].groupby([df_bool['week']==week-3 ,df_bool['v']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df_bool['times'].groupby([df_bool['week']==week-4 ,df_bool['v']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	f2=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	f2=np.array(f2.tolist()*num*DAY).reshape((num*V*DAY,5))
	return f2
	
def user_feature1(df,num,week):
	V_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['user']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['user']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df['times'].groupby([df['week']==week-1,df['day']==5,df['user']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df['times'].groupby([df['week']==week-1,df['day']==4,df['user']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df['times'].groupby([df['week']==week-1,df['user']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df['times'].groupby([df['week']==week-1 ,df['user']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df['times'].groupby([df['week']==week-2,df['user']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df['times'].groupby([df['week']==week-3 ,df['user']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df['times'].groupby([df['week']==week-4 ,df['user']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	f3=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	F=f3.copy()
	for i in range(DAY*V-1):
		F=np.concatenate((F,f3),axis=1)
	f3=F.reshape((num*V*DAY,5))
	return f3

def user_feature2(df,num,week):
	df_bool=df.copy()
	df_bool['times']=df['times']>0
	
	V_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['user']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['user']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==5,df_bool['user']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==4,df_bool['user']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['user']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df_bool['times'].groupby([df_bool['week']==week-1 ,df_bool['user']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df_bool['times'].groupby([df_bool['week']==week-2,df_bool['user']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df_bool['times'].groupby([df_bool['week']==week-3 ,df_bool['user']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df_bool['times'].groupby([df_bool['week']==week-4 ,df_bool['user']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	f4=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	F=f4.copy()
	for i in range(DAY*V-1):
		F=np.concatenate((F,f4),axis=1)
	f4=F.reshape((num*V*DAY,5))
	return f4
def user_feature3(df,num,week):	
	V_4_1=df['times'].groupby([df['week']==week-1 ,df['user']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_1=V_4_1.reshape((len(V_4_1),1))
	V_4_2=df['times'].groupby([df['week']==week-2,df['user']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_2=V_4_2.reshape((len(V_4_2),1))
	V_4_3=df['times'].groupby([df['week']==week-3 ,df['user']]).sum()  #last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_3=V_4_3.reshape((len(V_4_3),1))
	V_4_4=df['times'].groupby([df['week']==week-4,df['user']]).sum()  #last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	V_4_4=V_4_4.reshape((len(V_4_4),1))

	V_1=(V_4_1>0)
	V_2=(V_4_2>0)
	V_3=(V_4_3>0)
	V_4=(V_4_4>0)
	f3=np.concatenate((V_1,V_2,V_3,V_4),axis=1)
	F=f3.copy()
	for i in range(DAY*V-1):
		F=np.concatenate((F,f3),axis=1)
	f3=F.reshape((num*V*DAY,4))
	return f3
def user_v_feature1(df,num,week):
	V_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['user'],df['v']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df['times'].groupby([df['week']==week-1,df['day']==6,df['user'],df['v']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df['times'].groupby([df['week']==week-1,df['day']==5,df['user'],df['v']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df['times'].groupby([df['week']==week-1,df['day']==4,df['user'],df['v']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df['times'].groupby([df['week']==week-1,df['user'],df['v']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df['times'].groupby([df['week']==week-1 ,df['user'],df['v']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df['times'].groupby([df['week']==week-2,df['user'],df['v']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df['times'].groupby([df['week']==week-3 ,df['user'],df['v']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df['times'].groupby([df['week']==week-4 ,df['user'],df['v']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	f5=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	f5=f5.reshape((num,V,5))
	f5=np.concatenate((f5,f5,f5,f5,f5,f5,f5),axis=1).reshape((num*DAY*V,5))
	return f5

def user_v_feature2(df,num,week):
	df_bool=df.copy()
	df_bool['times']=df['times']>0
	
	V_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['user'],df_bool['v']]).sum()  #last day
	V_1=V_1.values[len(V_1)*3/4:]
	V_1=V_1.reshape((len(V_1),1))
	
	V_2_1=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==6,df_bool['user'],df_bool['v']]).sum()  #last 3 days
	V_2_1=V_2_1.values[len(V_2_1)*3/4:]
	V_2_2=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==5,df_bool['user'],df_bool['v']]).sum()  #last 3 days
	V_2_2=V_2_2.values[len(V_2_2)*3/4:]
	V_2_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['day']==4,df_bool['user'],df_bool['v']]).sum()  #last 3 days
	V_2_3=V_2_3.values[len(V_2_3)*3/4:]
	V_2=V_2_1+V_2_2+V_2_3
	V_2=V_2.reshape((len(V_2),1))
	
	V_3=df_bool['times'].groupby([df_bool['week']==week-1,df_bool['user'],df_bool['v']]).sum()  #last 7 days
	V_3=V_3.values[len(V_3)/2:]
	V_3=V_3.reshape((len(V_3),1))
	
	V_4_1=df_bool['times'].groupby([df_bool['week']==week-1 ,df_bool['user'],df_bool['v']]).sum()  #last week
	V_4_1=V_4_1.values[len(V_4_1)/2:]
	V_4_2=df_bool['times'].groupby([df_bool['week']==week-2,df_bool['user'],df_bool['v']]).sum()  #last last week
	V_4_2=V_4_2.values[len(V_4_2)/2:]
	V_4_3=df_bool['times'].groupby([df_bool['week']==week-3 ,df_bool['user'],df_bool['v']]).sum()  #last last last week
	V_4_3=V_4_3.values[len(V_4_3)/2:]
	V_4_4=df_bool['times'].groupby([df_bool['week']==week-4 ,df_bool['user'],df_bool['v']]).sum()  #last last last last week
	V_4_4=V_4_4.values[len(V_4_4)/2:]
	
	V_4=V_4_1+V_4_2
	V_4=V_4.reshape((len(V_4),1))
	
	V_5=V_4_1+V_4_2+V_4_3+V_4_4
	V_5=V_5.reshape((len(V_5),1))
	
	f6=np.concatenate((V_1,V_2,V_3,V_4,V_5),axis=1)
	f6=f6.reshape((num,V,5))
	f6=np.concatenate((f6,f6,f6,f6,f6,f6,f6),axis=1).reshape((num*DAY*V,5))
	return f6

def user_day_v_feature(df,week):
	f1_1=df['times'].groupby([df['week']==week-1,df['user'],df['day'],df['v']]).sum()
	f1_1=f1_1.values[len(f1_1)/2:]  #last week ,user,day ,v
	f1_1=f1_1.reshape((len(f1_1),1))
	
	f1_2=df['times'].groupby([df['week']==week-2,df['user'],df['day'],df['v']]).sum()
	f1_2=f1_2.values[len(f1_2)/2:]  #last last week ,user,day,v
	f1_2=f1_2.reshape((len(f1_2),1))
	
	f1_3=df['times'].groupby([df['week']==week-3,df['user'],df['day'],df['v']]).sum()
	f1_3=f1_3.values[len(f1_3)/2:]  #last last last week ,user,day,v
	f1_3=f1_3.reshape((len(f1_3),1))
	
	f1_4=df['times'].groupby([df['week']==week-4,df['user'],df['day'],df['v']]).sum()
	f1_4=f1_4.values[len(f1_4)/2:]  #last last last week ,user,day,v
	f1_4=f1_4.reshape((len(f1_4),1))
	
	f1=np.concatenate((f1_1,f1_2,f1_3,f1_4),axis=1)
	return f1

def no_act(df,num,week):
	f1_1=df['times'].groupby([df['week']==week-1,df['user'],df['v']]).sum()
	f1_1=f1_1.values[len(f1_1)/2:]  #last week ,user,day ,v

	
	f1_2=df['times'].groupby([df['week']==week-2,df['user'],df['v']]).sum()
	f1_2=f1_2.values[len(f1_2)/2:]  #last last week ,user,day,v

	
	f1_3=df['times'].groupby([df['week']==week-3,df['user'],df['v']]).sum()
	f1_3=f1_3.values[len(f1_3)/2:]  #last last last week ,user,day,v

	
	f1_4=df['times'].groupby([df['week']==week-4,df['user'],df['v']]).sum()
	f1_4=f1_4.values[len(f1_4)/2:]  #last last last week ,user,day,v

	
	f1= ((f1_1+f1_2+f1_3+f1_4)>0)

	f1=f1.reshape((num,V,1))
	f8=np.concatenate((f1,f1,f1,f1,f1,f1,f1),axis=1).reshape((num*DAY*V,1))
	
	return f8
def no_act_add(df,num,week):
	f1_1=df['times'].groupby([df['week']==week-1,df['user'],df['v']]).sum()
	f1_1=f1_1.values[len(f1_1)/2:]  #last week ,user,day ,v

	
	f1_2=df['times'].groupby([df['week']==week-2,df['user'],df['v']]).sum()
	f1_2=f1_2.values[len(f1_2)/2:]  #last last week ,user,day,v

	
	f1_3=df['times'].groupby([df['week']==week-3,df['user'],df['v']]).sum()
	f1_3=f1_3.values[len(f1_3)/2:]  #last last last week ,user,day,v

	
	f1_4=df['times'].groupby([df['week']==week-4,df['user'],df['v']]).sum()
	f1_4=f1_4.values[len(f1_4)/2:]  #last last last week ,user,day,v

	
	f1= (f1_1>0)
	f2= ((f1_1+f1_2)>0)
	f3= ((f1_1+f1_2+f1_3)>0)

	f1=f1.reshape((len(f1),1))
	f2=f2.reshape((len(f2),1))
	f3=f3.reshape((len(f3),1))
	f8=np.concatenate((f1,f2,f3),axis=1)
	f8=f8.reshape((num,V,3))
	f8=np.concatenate((f8,f8,f8,f8,f8,f8,f8),axis=1).reshape((num*DAY*V,3))

	return f8
	
	
# ###########useless feature############
def user_cluster(df,num,week):
	f1_1=df['times'].groupby([df['week']==week-1,df['user'],df['day'],df['v']]).sum()
	f1_1=f1_1.values[len(f1_1)/2:].reshape((num,DAY*V))  #last week ,user,day ,v

	
	f1_2=df['times'].groupby([df['week']==week-2,df['user'],df['day'],df['v']]).sum()
	f1_2=f1_2.values[len(f1_2)/2:].reshape((num,DAY*V))  #last last week ,user,day,v

	
	f1_3=df['times'].groupby([df['week']==week-3,df['user'],df['day'],df['v']]).sum()
	f1_3=f1_3.values[len(f1_3)/2:].reshape((num,DAY*V))  #last last last week ,user,day,v

	
	f1_4=df['times'].groupby([df['week']==week-4,df['user'],df['day'],df['v']]).sum()
	f1_4=f1_4.values[len(f1_4)/2:].reshape((num,DAY*V))  #last last last week ,user,day,v

	f=np.concatenate((f1_1,f1_2,f1_3,f1_4),axis=1).reshape((num,DAY*V*4))
	f=f*1.0
	f = preprocessing.scale(f)
	from sklearn import cluster
	model = cluster.KMeans(init='k-means++', n_clusters=20, n_init=10,n_jobs=5)
	model=model.fit(f)
	user_group = model.predict(f)
	user_group=user_group.reshape((num,1))
	enc = preprocessing.OneHotEncoder()
	enc.fit(np.arange(20).reshape((20,1)))
	del f1_1 ,f1_2,f1_3,f1_4,f
	user_onehot=enc.transform(user_group).toarray()
	f9=user_onehot.copy()
	for i in range(DAY*V-1):
		f9=np.concatenate((f9,user_onehot),axis=1)
	f9=f9.reshape((num*V*DAY,20))	
	return f9
def last_week_everyday(df,num,week):
	f10=df['times'].groupby([df['week']==week-1,df['user'],df['v'],df['day']]).sum()
	f10=f10.values[len(f10)/2:]  #last week ,user,day ,v
	f10=f10.reshape((num,V,DAY))
	f10=np.concatenate((f10,f10,f10,f10,f10,f10,f10),axis=1).reshape((num*V*DAY,DAY))
	return f10
def onehot():#70hot
	train_X=np.load('data/train_XX.npy')
	enc = preprocessing.OneHotEncoder()
	enc.fit(np.arange(70).reshape((70,1)))
	a=train_X[:,1]*10+train_X[:,2]
	hot=(enc.transform(a.reshape((len(a),1))).toarray())*1000
	train_X=np.concatenate((hot,train_X[:,3:]),axis=1)
	np.save('data/train_X_70hot.npy',train_X)
	
	test_X=np.load('data/test_XX.npy')
	enc = preprocessing.OneHotEncoder()
	enc.fit(np.arange(70).reshape((70,1)))
	a=test_X[:,1]*10+test_X[:,2]
	hot=(enc.transform(a.reshape((len(a),1))).toarray())*1000
	test_X=np.concatenate((hot,test_X[:,3:]),axis=1)
	np.save('data/test_X_70hot.npy',test_X)
