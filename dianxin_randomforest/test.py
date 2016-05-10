#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
from numpy import *
def get_result():
	reference=open('reference.txt','r')
	result=open('result.txt','r')

	reference_dict={}
	result_dict={}

	for line in reference.readlines():
		line=line.split('\t')
		dat=line[1].split(',')
		reference_dict[line[0]]=zeros(70)
		for i in range(len(dat)):
			reference_dict[line[0]][i]=int(dat[i].strip())
	reference.close()
	print 'read reference data successfully!'

	for line in result.readlines():
		line=line.split('\t')
		dat=line[1].split(',')
		result_dict[line[0]]=zeros(70)
		for i in range(len(dat)):
			result_dict[line[0]][i]=int(dat[i].strip())
	result.close()
	print 'read result data successfully!'

	UserCount=len(result_dict)
	rUserCount=len(reference_dict)
	hitUserCount=0

	SumSimilarity=0.0

	for elem in result_dict:
		Similarity=0.0
		if(reference_dict.has_key(elem)):
			a=1.0*sum(reference_dict[elem]*result_dict[elem])
			b=1.0*sqrt(sum(reference_dict[elem]*reference_dict[elem]))*sqrt(sum(result_dict[elem]*result_dict[elem]))
			if b==0:
				print elem
				break
			Similarity=a/b
			SumSimilarity+=Similarity
			hitUserCount+=1

	precision=SumSimilarity/UserCount
	recall=1.0*hitUserCount/rUserCount

	F1=2*precision*recall/(precision+recall)
	print 'precision=',precision*100,'%'
	print 'recall=',recall*100,'%'
	print 'F1=',F1*100,'%'

def get_score(y,yy):
	y=y+0.5
	y=y.astype(int)
	yy=yy+0.5
	yy=yy.astype(int)
	num = y.shape[0]/70
	y_tmp = y.reshape(num,70)
	yy_tmp = yy.reshape(num,70)
	pre = 0
	recall = 0
	for i in range(num):
		if y_tmp[i,:].sum()>0:
			if yy_tmp[i,:].sum()>0:
				recall += 1
				a = (y_tmp[i,:]*yy_tmp[i,:]).sum()
				b = ((y_tmp[i,:]**2).sum())**0.5
				c = ((yy_tmp[i,:]**2).sum())**0.5
				pre += a/b/c
	pre /= num
	recall /= num
	recall=1.0
	F1=pre*2*recall/(recall+pre)
	print 'precision=',pre*100,'%'
	print 'recall=',recall*100,'%'
	print 'F1=',F1*100,'%'
