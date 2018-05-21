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

class SingleGaussian:
    def __init__(self):
        self.vowels_d1 = {}
        self.vowels_d2 = {}
        self.prior = {}
        self.cond_prob = {}
        self.vowel_names = []

    def TrainConvSingleGaussian(self, train_data, train_labels):
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

        for vowel in prior_count.keys():
            self.prior[vowel] = prior_count[vowel] / float(num_data)
            self.cond_prob.setdefault(vowel, {})
            self.cond_prob[vowel].setdefault('means', np.zeros((2, 1)))
            self.cond_prob[vowel].setdefault('convar', np.zeros((2, 2)))
            self.cond_prob[vowel]['means'][0] = np.mean(vowels_d1[vowel])
            self.cond_prob[vowel]['means'][1] = np.mean(vowels_d2[vowel])
            matrix_cov = np.concatenate(
                (np.array(vowels_d1[vowel]).reshape(1, -1), np.array(vowels_d2[vowel]).reshape(1, -1)), axis=0)
            self.cond_prob[vowel]['convar'] = np.cov(matrix_cov)
        return

    def TestConvSingleGaussian(self, test_data, test_labels):
        num_test_data = len(test_data)
        acc_count = 0
        predict_labels = []
        true_labels = []
        f1 = open('Predict_ConvSingleGaussian.txt', 'a')
        for i, data in enumerate(test_data):
            max_post = 0
            predict_vowel = ''
            data_array = np.array([float(data[0]), float(data[1])]).reshape(-1, 1)
            for vowel in self.prior.keys():
                temp_post = self.prior[vowel] * Gaussian_2D(self.cond_prob[vowel]['means'],
                                                            self.cond_prob[vowel]['convar'], data_array)
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
    train = ReadData('train.txt')
    test = ReadData('test.txt')
    S = SingleGaussian()
    print "Convariance Single Gaussian Accuracy: "
    S.TrainConvSingleGaussian(train.data, train.labels)
    print S.TestConvSingleGaussian(test.data, test.labels)