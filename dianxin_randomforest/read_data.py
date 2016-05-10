import sys
import numpy as np
from pandas import DataFrame
import cPickle as pickle
import time
V=10
WEEK=7
DAY=7
threshold=50
def prepare():
	dat=open('data/part-r-00000','r')
	dat_write=open('data/part-r-00000_nihao','w')
	count=0
	seg='\t'
	map={}#(user,user_id)
	map_dic={}  #(user_id,user)
	dic={}
	for line in dat.readlines():
		line=line.split('\t')
		#print (v,day,times)
		if(not map.has_key(line[0])):
			map[line[0]]=str(count)
			dic[str(count)]=[]
			count+=1
		line[0]=map[line[0]]
		dic[line[0]].append(seg.join(line))
	for i in range(len(dic)):
		for j in range(len(dic[str(i)])):
			dat_write.write(dic[str(i)][j])
	dat_write.close()
	for user in map:
		map_dic[int(map[user])]=user
	dat.close()
	map_file=open('data/map.pkl','wb')
	pickle.dump(map_dic,map_file)
	map_file.close()
	map_user_ID_file=open('data/map_user_ID.pkl','wb')
	pickle.dump(map,map_user_ID_file)
	map_user_ID_file.close()
		

'''
def read_dat2(num):
	start_time=time.time()
	dic={}
	print 'read data begining'
	dat=open('part-r-00000_nihao','r')
	for line in dat.readlines():
		line=line.split('\t')
		if int(line[0])>=num:
			break
		week=int(line[1][1])-1
		day=int(line[1][-1])-1
		v=int(line[2][1:])-1
		times=int(line[3].strip())
		if times>threshold:
			times=threshold
		#print (v,day,times)
		if(not dic.has_key(line[0])):
			dic[line[0]]=np.zeros((V,WEEK*DAY))
			dic[line[0]]=dic[line[0]].tolist()
		dic[line[0]][v][week*WEEK+day]=times
	dat.close()
	mm=DataFrame(dic)
	print 'read data successfully!!!'
	print 'read data took %fs!' % (time.time() - start_time)
	return mm.sort_index(axis=1)
'''
def read_dat(num):
	start_time=time.time()
	print 'read data begining'
	dat=open('data/part-r-00000_nihao','r')
	data=np.zeros((num*WEEK*DAY*V,5))
	count=0
	for user in range(num):
		for v in range(V):
			for week in range(WEEK):
				for day in range(DAY):
					data[count][0]=user
					data[count][1]=v
					data[count][2]=week
					data[count][3]=day
					count+=1
	for line in dat.readlines():
		line=line.split('\t')
		user_id=int(line[0])
		if user_id>=num:
			break
		week=int(line[1][1])-1
		day=int(line[1][3])-1
		v=int(line[2][1:])-1
		times=int(line[3].strip())
		if times>threshold:
			times=threshold
		data[user_id*490+v*49+week*7+day][4]+=times
	dat.close()
	df=DataFrame(data,columns=['user','v','week','day','times'])
	print 'read data successfully!!!'
	print 'read data took %fs!' % (time.time() - start_time)
	np.save('data/df.npy',df)
	return df	
	
def read_dat2():
	start_time=time.time()
	dic=[]
	print 'read data begining'
	dat=open('part-r-00000_nihao','r')
	for line in dat.readlines():
		line=line.split('\t')
		week=int(line[1][1])-1
		day=int(line[1][-1])-1
		v=int(line[2][1:])-1
		times=int(line[3].strip())
		if times>threshold:
			times=threshold
		dic.append([int(line[0]),v,week,day,times])
	dat.close()
	mm=DataFrame(dic,columns=['user','v','week','day','times'])
	print 'read data successfully!!!'
	print 'read data took %fs!' % (time.time() - start_time)
	return mm


