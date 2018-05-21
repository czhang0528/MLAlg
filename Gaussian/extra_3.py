#!/usr/bin/evn Python
# coding=utf-8
"""
Author: Cheng Zhang
Contact: zhang.7804@osu.edu
Date: 4/13/2017
Description: Gaussian
"""

import sys
import numpy as np

import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class ReadData:
    def __init__(self, fileName):
        self.data = []
        self.labels = []
        for line in open(fileName):
            line = line.split()
            self.data.append(line[0:-1])
            self.labels.append(line[-1])

def Gaussian_1D(mean, std, x):
    return (np.exp(-1 * (x - mean) ** 2 / float(2 * std ** 2)) / float(np.sqrt(2 * np.pi) * std))

def Gaussian_2D(means, covar, xs):
    k = len(xs)
    return np.exp(-0.5 * (xs - means).T.dot(np.linalg.pinv(covar)).dot(xs - means)) / float(
        np.sqrt((2 * np.pi) ** k * np.linalg.det(covar)))

class MixtureOfGaussian:
    def __init__(self, num_gaussian, num_iter):
        self.num_gaussian = num_gaussian
        self.num_iter = num_iter
        self.vowels_d1 = {}
        self.vowels_d2 = {}
        self.prior = {}
        self.cond_prob = {}
        self.cond_prob_cov = {}
        self.vowel_names = []


    def EM_Conv(self, train_2d_array, vowel):
        # initialization
        dims = train_2d_array.shape[0]
        num_data = train_2d_array.shape[1]
        prior = np.random.rand(self.num_gaussian)
        prior = [x / float(sum(prior)) for x in prior]
        means = []
        convars = []
        for i in range(self.num_gaussian):
            mean = np.random.rand(dims)
            mean = np.array([np.mean(train_2d_array[j, :]) - mean[j] for j in range(dims)]).reshape(-1, 1)
            means.append(mean)
            convar = np.random.rand(dims, dims)
            convar = np.cov(train_2d_array) - convar
            convars.append(convar)
        log_likelihood = []
        current_log = 0
        previous_log = 0
        for it in range(self.num_iter):
            temp_prob = 0
            for idx in range(num_data):
                temp_prob += -1 * np.log(
                    sum([prior[idx_g] * Gaussian_2D(means[idx_g], convars[idx_g], train_2d_array[:, idx].reshape(-1, 1)) \
                         for idx_g in range(self.num_gaussian)]))
            log_likelihood.append(float(temp_prob) / float(num_data))

            current_log = temp_prob / float(num_data)
            if abs(current_log - previous_log) < 10 ** -10:
                break

            prior_count = np.zeros((self.num_gaussian))
            cond_prob = np.zeros((num_data, self.num_gaussian))
            # E-step
            for j in range(num_data):
                data_array = train_2d_array[:, j].reshape(-1, 1)
                temp_cond_prob = [Gaussian_2D(means[i], convars[i], data_array) * prior[i] for i in
                                  range(self.num_gaussian)]
                cond_prob[j, :] = np.array([x / float(sum(temp_cond_prob)) for x in temp_cond_prob]).reshape(1, -1)
                prior_count = np.array([prior_count[i] + cond_prob[j, i] for i in range(self.num_gaussian)])
            # M-step
            for i in range(self.num_gaussian):
                for k in range(dims):
                    means[i][k] = sum([cond_prob[j, i] * float(train_2d_array[k, j]) for j in range(num_data)]) / float(
                        prior_count[i])
                t = (train_2d_array - means[i])
                convars[i] = (cond_prob[:, i] * t).dot(t.T) / float(prior_count[i])
                prior[i] = prior_count[i] / float(sum(prior_count))
            previous_log = current_log

        return prior, means, convars

    def TrainConvMoG(self, train_data, train_labels):
        num_data = len(train_data)
        prior_count = {}
        vowels_d1 = {}
        vowels_d2 = {}
        for i, data in enumerate(train_data):
            prior_count.setdefault(train_labels[i], 0)
            vowels_d1.setdefault(train_labels[i], [])
            vowels_d2.setdefault(train_labels[i], [])
            prior_count[train_labels[i]] += 1
            vowels_d1[train_labels[i]].append(float(data[0]))
            vowels_d2[train_labels[i]].append(float(data[1]))
        self.vowel_names = prior_count.keys()

        for vowel in prior_count.keys():
            self.prior[vowel] = prior_count[vowel] / float(num_data)
            self.cond_prob_cov.setdefault(vowel, [])
            vowels_data = np.concatenate(
                (np.array(vowels_d1[vowel]).reshape(1, -1), np.array(vowels_d2[vowel]).reshape(1, -1)), axis=0)
            MoG_prior_d1, MoG_means_d1, MoG_stds_d1 = self.EM_Conv(vowels_data, vowel)
            for M_p, M_m, M_s in zip(MoG_prior_d1, MoG_means_d1, MoG_stds_d1):
                self.cond_prob_cov[vowel].append([M_p, M_m, M_s])
        # print self.cond_prob
        return

    def TestConvMoG(self, test_data, test_labels):
        num_test_data = len(test_data)
        acc_count = 0
        predict_labels = []
        true_labels = []
        f1 = open('Predict_ConvMistureOfGaussian.txt', 'a')
        for i, data in enumerate(test_data):
            max_post = 0
            predict_vowel = ''
            data_array = np.array([float(data[0]), float(data[1])]).reshape(-1, 1)
            for vowel in self.prior.keys():
                temp_cond_prob = 0
                for j in range(self.num_gaussian):
                    temp_cond_prob += self.cond_prob_cov[vowel][j][0] * \
                                      Gaussian_2D(self.cond_prob_cov[vowel][j][1], self.cond_prob_cov[vowel][j][2],
                                                  data_array)
                temp_post = self.prior[vowel] * temp_cond_prob
                if temp_post > max_post:
                    max_post = temp_post
                    predict_vowel = vowel
            if predict_vowel == test_labels[i]:
                acc_count += 1
            predict_labels.append(predict_vowel)
            true_labels.append(test_labels[i])
            f1.write(predict_vowel + ' ' + test_labels[i] + '\n')
        f1.close()

        return acc_count / float(num_test_data)

if __name__ == '__main__':
    train = ReadData('/train.txt')
    test = ReadData('test.txt')
    M = MixtureOfGaussian(int(sys.argv[1]), int(sys.argv[2]))
    print "Convariance MoG Accuracy: "
    M.TrainConvMoG(train.data, train.labels)
    print M.TestConvMoG(test.data, test.labels)