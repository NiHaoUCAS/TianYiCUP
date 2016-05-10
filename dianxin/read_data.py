import sys
from numpy import *
import time
V=10
WEEK=7
DAY=7
threshold=50
def read_dat():
	start_time=time.time()
	print 'read data begining'
	dic={}
	dat=open('part-r-00000','r')
	for line in dat.readlines():
		line=line.split('\t')
		week=int(line[1][1])-1
		day=int(line[1][-1])-1
		v=int(line[2][1:])-1
		times=int(line[3].strip())
		if times>threshold:
			times=threshold
		#print (v,day,times)
		if(dic.has_key(line[0])):
			dic[line[0]][week][v][day]=times
		else :
			dic[line[0]]=zeros((WEEK,V,DAY))
			dic[line[0]][week][v][day]=times
	dat.close()
	record=open('recod.txt','w')
	for user in dic:
		record.write('\n'+user+'\t')
		for week in range(WEEK):
			record.write('\t')
			for day in range(DAY):
				record.write(str(int(dic[user][week][0][day]))+',')
	record.close()
	print 'read data successfully!!!'
	print 'read data took %fs!' % (time.time() - start_time)
	return dic


