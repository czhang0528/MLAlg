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
		label_C.append(int(line[0]))
		feature.append(map(float,instance[1:]))
	in_file.close()
	return feature,label_C

def compute_euc_dist(point_1, point_2):
	return sqrt(sum(power(mat(point_1) - mat(point_2), 2)))	

# k-means clustering
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
			#print i
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

		# step 3 : update the centroids
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
	print numSample
	colors=['b+','r+','g+','m+','c+','y*','g*','m*','c*','r*']
	for i in xrange(numSample): 
		markIndex = int(cluster[i][0])
		plt.plot(data[i][0], data[i][1], colors[markIndex])

	mark=['bD','rD','gD','mD','cD','yD','gD','mD','cD','rD']
	for i in range(k):
		plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 8, markeredgecolor='black')
	plt.show()

def plot_original_points(data, label):
	numSample = shape(data)[0]
	print numSample
	dim = shape(data)[1]
	if dim != 2:
		return 1
	colors=['b+','r+','g+','m+','c+','y*','g*','m*','c*','r*']
	for i in xrange(numSample): 
		markIndex = label[i] # for convert_continuous_to_discrete
		plt.plot(data[i][0], data[i][1], colors[markIndex])
	plt.show()

# step 2 : compute prior P(C)
def compute_prior(feature, label, alpha):
	prior = []
	category = list(set(label))
	category.sort()
	for c in category:
		quantity_c = label.count(c)
		prior_c = (quantity_c + alpha)/(float(len(label)) + len(category) * alpha)
		prior.append(prior_c)
	return prior

# step 2 : compute conditional probabilities P(V|C)
def compute_cond_prob(feature, label, alpha, centroids, cluster, k):
	cond_prob = {}
	dim = shape(feature)[1]
	category = list(set(label))
	category.sort()
	multilist = [[0 for col in range(len(category))] for row in range(k)]
	for c in category: #c = '0','1','2','3','4','5','6'
		quantity_c = label.count(c)
		for index in range(len(cluster)): #index = 0 ... 499
			if (label[index] == c):
				for i in range(k): # i = 0 ... 9
					if (cluster[index][0] == i):
						multilist[i][int(c)] = multilist[i][int(c)] + 1
	for c in category:
		for i in range(k):
			multilist[i][int(c)] = (multilist[i][int(c)] + alpha) / (float(label.count(c)) + dim * alpha)
	return multilist

def convert_continuous_to_discrete(data, centroids, k):
	numSample = shape(data)[0]
	discrete_label = [0] * numSample
	for i in range(numSample):
		minDist = 1000000
		for j in range(k):
			dist = compute_euc_dist(centroids[j],data[i])
			if dist < minDist:
				minDist = dist
				discrete_label[i] = j
	return discrete_label

def bayes_test(data,prior,cond_prob,discrete_label):
	numSample = shape(data)[0]
	category = len(prior)
	predict = []
	for i in range(numSample):
		predict_label = -1
		p = -1
		for c in range(category):
			cnt_index = discrete_label[i]
			cmp_p = cond_prob[cnt_index][c] * prior[c]
			if cmp_p > p:
				p = cmp_p
				predict_label = c
		predict.append(predict_label)
	return predict
	
def evaluate(predict,gt_label):
	count = 0
	for i in range(len(predict)):
			if predict[i] == int(gt_label[i]):
				count = count + 1
	error_rate = 100 - count * 100 / float(len(gt_label))
	return error_rate


runs = 1
candidate = [10]
for k in candidate:
	for i in range(runs):
		alpha = 0
		feature_train, label_train = load_data(sys.argv[1])
		feature_test, label_test = load_data(sys.argv[2])
		centroids, cluster = kmeans(feature_train,k)
		discrete_label = convert_continuous_to_discrete(feature_test, centroids, k)
		prior = compute_prior(feature_train, label_train, alpha)
		print 'Prior:'
		print prior
		cond_prob = compute_cond_prob(feature_train, label_train, alpha, centroids, cluster, k)
		print 'Conditional Probabilities:'
		print cond_prob
		#predict = bayes_test(feature_test,prior,cond_prob,discrete_label)
		#error_rate = evaluate(predict,label_test)
		#print error_rate



