#!/usr/bin/env Python
#coding=utf-8
"""
Author: Cheng Zhang
Contact: zhang.7804@osu.edu
Date: 1/27/2017
Description: Naive Bayes algorithm on mushroom dataset
			 The script is also suitable for multi-class classification with multiple feature dimensions 
"""
import sys
import time

class Naive_Bayes_Multinominal(object):
	"""
	Naive Bayes classifier for multinomial models
	Attributes
	----------
	train(trainfile) : trainfile is the path of training dataset
	test(testfile):    testfile is the path of testing dataset
	evaluate(outfile): outfile is the output path of predict label 

	"""
	def __init__(self,alpha = 1):
		self.train_label = list()
		self.label = list() #list of labels
		self.feature = list() #list of features
		self.category = None #list() of category
		self.category_prior = None #list of each category's prior (1 * category_number)
		self.category_prior_dic = None #map list
		self.conditional_prob = None #dictionary of each feature's conditional probability ()
		self.conditional_prob_dic = None #map list
		self.alpha = alpha #smooth value, default = 1.0
		self.predict_label_list = None

	def _load_dataset(self,filename):
		in_file = open(filename,'r')
		for line in in_file.readlines():
			instance = line.strip().split(',')
			self.label.append(line[0])
			self.train_label.append(line[0])
			self.feature.append(instance[1:])
		in_file.close()

	def train(self,trainfile):
		self._load_dataset(trainfile)
		# calculate prior probabilities for each category: p(y_i)
		start = time.time()
		self.category = list(set(self.label)) #remove duplicate labels in the label list from training set
		self.category_prior = [] #set prior probabilities list to empty
		self.category_prior_dic = []
		for c in self.category:
			quantity_c = self.label.count(c)
			self.category_prior_dic.append(c)
			self.category_prior.append((quantity_c + self.alpha)/float(len(self.label)) + len(self.category) * self.alpha)

		# calculate conditional probabilities for each feature: p(x_j|y_i)
		self.conditional_prob = {} #set conditional probability dictionary to empty
		self.conditional_prob_dic = {}
 		for c in self.category: # labels
 			self.conditional_prob[c] = {}
 			self.conditional_prob_dic[c] = {}
 			feature_c = []
 			for index_c in range(len(self.label)):
 				if(self.label[index_c] == c):
 					feature_c.append(self.feature[index_c]) #feature_c is (number_c * feature_dim)

 			for index_i in range(len(self.feature[0])): # index from 0~21
 				feature_i = []
 				for label_index in range(len(feature_c)): # index from 0~2989(p),0~3133(e)
 					feature_i.append(feature_c[label_index][index_i])
 				feature_i_unique = list(set(feature_i))
 				self.conditional_prob_dic[c][index_i] = feature_i_unique
 				cond_prob_temp_list = []
 				for feature_value in feature_i_unique:
					cond_prob_temp_list.append((feature_i.count(feature_value) + self.alpha)/(float(len(feature_c)) + len(self.feature[0]) * self.alpha))
 					self.conditional_prob[c][index_i] =  cond_prob_temp_list
	
	def test(self,testfile):
		self.label = [] #list of labels
		self.feature = [] #list of features
		self._load_dataset(testfile)
		self.predict_label_list = []
		for instance in range(len(self.feature)):
			predict = {}
			for category_index in self.category_prior_dic:
				predict[category_index] = {}
				try:
					temp_prob = self.category_prior[self.category_prior_dic.index(category_index)]
				except ValueError:
					pass
				for feature_index in range(len(self.feature[0])):
					try:
						cond_prob_dic_index = self.conditional_prob_dic[category_index][feature_index].index(self.feature[instance][feature_index])
						temp_prob = temp_prob * self.conditional_prob[category_index][feature_index][cond_prob_dic_index]
					except ValueError:
						temp_prob = temp_prob * ((0 + self.alpha) / (self.train_label.count(category_index) + len(self.feature[0]) * self.alpha))		
				predict[category_index] = temp_prob
			predict_sort = sorted (predict.iteritems(), key=lambda d:d[1], reverse=True)
			self.predict_label_list.append(predict_sort[0][0])
	
	def evaluate(self,outfile):
		count = 0
		for i in range(len(self.label)):
			if self.predict_label_list[i] == self.label[i]:
				count = count + 1
		accuracy = count * 100 / float(len(self.label))
		f = open(outfile,'w')
		for i in self.predict_label_list:
			f.write(i)
			f.write("\n")
		f.close()
		return accuracy

def main():
		
	nb = Naive_Bayes_Multinominal()

	train_time = time.time()
	nb.train(sys.argv[1])
	train_time = time.time() - train_time

	test_time = time.time()
	nb.test(sys.argv[2])
	test_time = time.time() - test_time

	accuracy = nb.evaluate(sys.argv[3])
	print ' Accuracy: %r %%' % (accuracy)
	print ' Training time: %.4f sec' % (train_time)
	print ' Testing time: %.4f sec' % (test_time)

if __name__=='__main__':
	main()