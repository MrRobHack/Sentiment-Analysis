import math
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kMeans(sentimentList):
	# Parameters k = number of clusters, & max iterations
	k = 3
	maxIter = 2500
	# Randomly selecting three unique centers
	centroid = random.sample(sentimentList, k)

	n = len(sentimentList)
	clusterLabels = [0] * n

	# Creating a list length to store number of data belonging to each cluster
	length = []
	
	while maxIter:
		# Assigning label of nearest centroid to each point
		for i in range(n):
			minDistance = math.dist(sentimentList[i], centroid[0]) # Euclidean distance
			clusterLabels[i] = 0
		for j in range(1, k):
			if math.dist(sentimentList[i], centroid[j]) < minDistance:# Euclidean
				minDistance = math.dist(sentimentList[i], centroid[j])
				clusterLabels[i] = j
	
		#Creating a list for new centroid points
		newCentroid = []

		length= [0]*k
		for i in range(k):
			newCentroid.append([0.0, 0.0,0.0])


		# To find mean, first finding sum of data belonging to each cluster
		for i in range(n):
			temp = clusterLabels[i]
			newCentroid[temp][0] += sentimentList[i][0]
			newCentroid[temp][1] += sentimentList[i][1]
			newCentroid[temp][2] += sentimentList[i][2]
			length[temp] += 1
		
		# Now dividing sum by length of respective cluster
		for i in range(k):
			if length[i]==0:
				continue
			newCentroid[i][0] = newCentroid[i][0] / length[i]
			newCentroid[i][1] = newCentroid[i][1] / length[i]
			newCentroid[i][2] = newCentroid[i][2] / length[i]
		
		# Assigning new centroids to original centroids
		centroid = newCentroid
		maxIter -= 1

	# Returning labels belonging to each entry
	return clusterLabels,length[0],length[1],length[2]

def visual(len_c1,len_c2,len_c3):
    print("Number of elements in Cluster-1 :- "+str(len_c1))
    print("Number of elements in Cluster-2 :- "+str(len_c2))
    print("Number of elements in Cluster-3 :- "+str(len_c3))
    
    #Pi - chart data
    data = [len_c3,len_c1,len_c2]
    #label = ["Cluster-3: "+str(len_c3),"Cluster-1: "+str(len_c1),"Cluster-2: "+str(len_c2)]
    label = ["Cluster-3","Cluster-1","Cluster-2"]
    wp = { 'linewidth' : 0.1, 'edgecolor' : "green" }
    explode=(0.0,0.2,0.4)
    colors = ("cyan","orange","grey")
    def func(pct, allvalues):
        #print(pct)
        absolute = math.ceil(pct / 100*np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    fig = plt.figure(figsize =(10, 7))
    plt.pie(data, labels = label,
    		autopct = lambda pct: func(pct,data), 
    		shadow = True, 
    		wedgeprops = wp, 
    		explode = explode,
    		colors = colors, 
    		textprops= dict(color ="magenta",size= 8, weight="bold"))
    plt.title("Different Emotions Pie-Chart")
    plt.show()


def scatterplot(dataset):
	colors= ['blue','green','red']
	dataset['color']= dataset.Cluster.map({0:colors[0],1:colors[1],2:colors[2]})
	fig = plt.figure(figsize=(26,7))
	ax = fig.add_subplot(131, projection='3d')
	ax.scatter(dataset.Negative, dataset.Neutral, dataset.Positive, c=dataset.color)

	ax.set_xlabel('Negative')
	ax.set_ylabel('Neutral')
	ax.set_zlabel('Positive')
	ax.set_title('Cluster-Graph')
	plt.show()

	plt.scatter(dataset.Negative, dataset.Positive, c=dataset.color)
	plt.xlabel('Negative')
	plt.ylabel('Positive')
	plt.title('Cluster-plot Neg-Pos axis')
	plt.show()

	plt.scatter( dataset.Neutral, dataset.Positive, c=dataset.color)
	plt.xlabel('Neutral')
	plt.ylabel('Positive')
	plt.title('Cluster-plot Neu-Pos axis')
	plt.show()

	plt.scatter(dataset.Negative, dataset.Neutral,c=dataset.color)
	plt.xlabel('Negative')
	plt.ylabel('Neutral')
	plt.title('Cluster-plot Neg-Neu axis')
	plt.show()

if __name__ == "__main__":
	fp = open("data.txt", 'r')
	lines = fp.readlines() #in order of negative,neutral,positive
	fp.close()

	sentimentList,neg,neu,pos = [],[],[],[]
	for line in lines:
		temp = line.split(',')
		temp[2] = temp[2].rstrip('\n')
		temp[0] = float(temp[0])
		neg.append(float(temp[0]))
		temp[1] = float(temp[1])
		neu.append(float(temp[1]))
		temp[2] = float(temp[2])
		pos.append(float(temp[2]))
		sentimentList.append(temp)
		
	result, len_c1, len_c2, len_c3 = kMeans(sentimentList)
	dataset = pd.DataFrame(list(zip(neg,neu,pos,result)),
                      columns= ['Negative','Neutral','Positive','Cluster'])
	visual(len_c1,len_c2,len_c3)
	scatterplot(dataset)
