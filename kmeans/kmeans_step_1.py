#!/usr/bin/env Python
#coding=utf-8
"""
Author: Cheng Zhang
Contact: zhang.7804@osu.edu
Date: 2/12/2017
Description: K-means algorithm for homework #2
"""
from numpy import *
import matplotlib.pyplot as plt
import sys

def load_data(filename):
	label_C = [] 
	feature = []
	in_file = open(filename,'r')
	for line in in_file.readlines():
		instance = line.strip().split('\t')
		label_C.append(line[0])
		feature.append(map(float,instance[1:]))
	in_file.close()
	return feature,label_C

def compute_euc_dist(point_1, point_2):
	return sqrt(sum(power(mat(point_1) - mat(point_2), 2)))	

def kmeans(data, k):
	# step 1 : initialization
	numSample = shape(data)[0]
	dim = shape(data)[1]
	centroids = []
	for i in range(k):
		index = random.randint(0, numSample - 1)
		centroids.append(data[index])

	termination = True
	cluster = [[0,0]] * numSample

	while termination:
		termination = False
		# step 2 : find the nearest centroid for each sample
		for i in range(numSample):
			minDist = 1000000
			minIndex = -1
			for j in range(k):			
				dist = compute_euc_dist(centroids[j],data[i]) 
				if dist < minDist:
					minDist = dist
					minIndex = j
			if cluster[i][0] != minIndex:
				termination = True
			cluster[i] = [minIndex, minDist*minDist]
		for i in range(k):
			sample_in_cluster_i = []
			for j in range(numSample):
				if cluster[j][0] == i:
					sample_in_cluster_i.append(data[j])
			centroids[i] = mean(sample_in_cluster_i, axis=0).tolist()
	return centroids, cluster

def plot_cluster_points(data, k, centroids, cluster):
	numSample = shape(data)[0]
	dim = shape(data)[1]
	if dim != 2:
		return 1
	colors=['b+','r+','g+','m+','c+','y*','g*','m*','c*','r*']
	for i in xrange(numSample): 
		markIndex = int(cluster[i][0])
		plt.plot(data[i][0], data[i][1], colors[markIndex])

	mark=['bD','rD','gD','mD','cD','yD','gD','mD','cD','rD']
	for i in range(k):
		plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 8, markeredgecolor='black')

	plt.show()

def plot_original_points(data,label):
	numSample = shape(data)[0]
	dim = shape(data)[1]
	if dim != 2:
		return 1
	colors=['b+','r+','g+','m+','c+','y*','g*','m*','c*','r*']
	for i in xrange(numSample): 
		markIndex = int(label[i][0])
		plt.plot(data[i][0], data[i][1], colors[markIndex])
	plt.show()

k = 10
feature, label = load_data(sys.argv[1]) 
centroids, cluster = kmeans(feature,k)
plot_cluster_points(feature, k, centroids, cluster)


