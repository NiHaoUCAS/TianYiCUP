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
