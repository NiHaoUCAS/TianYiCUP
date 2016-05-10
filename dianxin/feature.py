import sys
from numpy import *
V=10
WEEK=7
DAY=7

dic={}
dat=open('part-r-00000','r')
for line in dat.readlines():
    line=line.split('\t')
    week=int(line[1][1])-1
    day=int(line[1][-1])-1
    v=int(line[2][1:])-1
    times=int(line[3].strip())
    #print (v,day,times)
    if(dic.has_key(line[0])):
        dic[line[0]][week][v][day]=times
    else :
        dic[line[0]]=zeros((WEEK,V,DAY))
        dic[line[0]][week][v][day]=times
dat.close()


result=open('result.txt','w')
for elem in dic:
    if(sum(dic[elem][WEEK-2])!=0):
        result.write(elem+'\t')
        for i in range(DAY):
            for j in range(V):
                if((i!=(DAY-1)) or (j!=(V-1))):
                    result.write(str(int(dic[elem][WEEK-2][j][i]))+',')
                else:
                    result.write(str(int(dic[elem][WEEK-2][j][i]))+'\n')
            
            
result.close()
print 'done'
    

