#!/usr/bin/evn Python
# coding=utf-8
"""
Author: Cheng Zhang
Contact: zhang.7804@osu.edu
Date: 4/13/2017
Description: Single Diagonal Gaussian
"""

import sys
import numpy as np

def Gaussian_1D(mean, std, x):
    return (np.exp(-1 * (x - mean) ** 2 / float(2 * std ** 2)) / float(np.sqrt(2 * np.pi) * std))

def Gaussian_2D(means, covar, xs):
    k = len(xs)
    return np.exp(-0.5 * (xs - means).T.dot(np.linalg.pinv(covar)).dot(xs - means)) / float(
        np.sqrt((2 * np.pi) ** k * np.linalg.det(covar)))

class load_data:
    def __init__(self, fileName):
        self.data = []
        self.labels = []
        for line in open(fileName):
            line = line.split()
            self.data.append(line[0:-1])
            self.labels.append(line[-1])

class Gaussian:
    def __init__(self):
        self.vowels_d1 = {}
        self.vowels_d2 = {}
        self.prior = {}
        self.cond_prob = {}
        self.vowel_names = []

    def Train(self, train_data, train_labels):
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
            self.cond_prob.setdefault(vowel, {})
            self.cond_prob[vowel].setdefault('mean_1', 0)
            self.cond_prob[vowel].setdefault('std_1', 0)
            self.cond_prob[vowel].setdefault('mean_2', 0)
            self.cond_prob[vowel].setdefault('std_1', 0)
            self.cond_prob[vowel]['mean_1'] = np.mean(vowels_d1[vowel])
            self.cond_prob[vowel]['std_1'] = np.std(vowels_d1[vowel])
            self.cond_prob[vowel]['mean_2'] = np.mean(vowels_d2[vowel])
            self.cond_prob[vowel]['std_2'] = np.std(vowels_d2[vowel])
        return

    def Test(self, test_data, test_labels):
        num_test_data = len(test_data)
        acc_count = 0
        predict_labels = []
        true_labels = []
        f1 = open('result_single_diag.txt', 'w')
        for i, data in enumerate(test_data):
            max_post = 0
            predict_vowel = ''
            for vowel in self.prior.keys():
                temp_post = self.prior[vowel] * Gaussian_1D(self.cond_prob[vowel]['mean_1'], self.cond_prob[vowel]['std_1'], float(data[0])) * Gaussian_1D(self.cond_prob[vowel]['mean_2'], self.cond_prob[vowel]['std_2'], float(data[1]))
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
    train = load_data('train.txt')
    test = load_data('test.txt')
    print "Accuracy: "
    S = Gaussian()
    S.Train(train.data, train.labels)
    print S.Test(test.data, test.labels)