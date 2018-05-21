#!/usr/bin/env Python
#coding=utf-8
"""
Author: Cheng Zhang
Contact: zhang.7804@osu.edu
Date: 2/24/2017
Description: AdaBoost algorithm
Reference: https://github.com/jakezhaojb/adaboost-py
"""

from numpy import *
import math,sys

def load_data(filename):
    label = []
    feature = []
    in_file = open(filename)
    for line in in_file:
        instance = line.strip().split(',')
        col = len(instance)
        label.append(int(instance[col-1]))
        feature.append(map(int,instance[0:col-1]))
    return feature,label

def stump(data,label):
    dataMat = mat(data)
    labelMat = mat(label).T
    numSample, dims = shape(dataMat)
    dataMat = dataMat.T
    dataMat = dataMat.tolist()
    # calculate entropy of a node prior to a split
    category_prior = []
    category_prior_dic = []
    category = list(set(label))
    for c in category:
        quantity_c = label.count(c)
        category_prior_dic.append(c)
        category_prior.append(quantity_c / float(len(label)))
    entropy_stump = 0
    for i in range(len(category)):
        entropy_stump = entropy_stump - category_prior[i] * math.log(category_prior[i],2)
    gain_dims = []
    for row in range(dims):
        gain = entropy_stump
        ct1 = 0
        ct2 = 0
        indiv = list(set(dataMat[row]))
        indiv_prior = []
        for k in indiv:
            quantity_indiv = indiv.count(k)
            indiv_prior.append(quantity_indiv / float(len(label)))
        tmp_entropy = 0
        for i in indiv: # check each dims
            ent = 0 
            for col in range(numSample):     
                if dataMat[row][col] == indiv[i]:
                    if label[col] == 1:
                        ct1 = ct1 + 1
                    if label[col] == -1:
                        ct2 = ct2 + 1
            ratio1 = ct1 / indiv.count(i)
            ratio2 = ct2 / indiv.count(i)
            ent = ent - ratio1 * math.log(ratio1,2) - ratio2 * math.log(ratio2,2)
            indiv_prior = indiv.count(i) / float(len(label))
            tmp_entropy = tmp_entropy + indiv_prior * ent
            gain = gain - tmp_entropy
        gain_dims.append(gain)
        print gain_dims

def stumpClassify(dataMat,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMat)[0],1))
    if threshIneq == 'lt':
        retArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(data,label,D):
    dataMat = mat(data)
    labelMat = mat(label).T
    m,n = shape(dataMat)

    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf 
    for i in range(n): # all dimensions
        rangeMin = dataMat[:,i].min(); rangeMax = dataMat[:,i].max();
        
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']: #less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                
                predictedVals = stumpClassify(dataMat,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['best_dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    print bestStump
    return bestStump,minError,bestClasEst
 
def train(dataArr,classLabels,numIt):
    weakClassArr = []  
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m) 
    aggClassEst = mat(zeros((m,1))) 
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha # weight
        weakClassArr.append(bestStump)

        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum() # normalization
        aggClassEst += alpha*classEst

        errorRate = 1.0*sum(sign(aggClassEst)!=mat(classLabels).T)/m
        if errorRate == 0.0:
            break
    return weakClassArr

def test(dataToClass,classifierArr):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)): 
        classEst = stumpClassify(dataMat,classifierArr[i]['best_dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return sign(aggClassEst)

def evaluate(predict,label):
    count_1 = 0
    count_m1 = 0
    total_1 = 0
    total_m1 = 0
    for i in range(len(predict)):        
        if int(predict[i]) == 1:
            total_1 = total_1 + 1
        if int(predict[i]) == -1:
            total_m1 = total_m1 + 1
        if int(predict[i]) == int(label[i]) == 1:
            count_1 = count_1 + 1
        if int(predict[i]) == int(label[i]) == -1:
            count_m1 = count_m1 + 1
    return (float(count_1) / total_1 + float(count_m1) / total_m1) / 2.0

if __name__=='__main__':
    train_file = sys.argv[1]#"../data/game_codedata_train.dat"
    test_file = sys.argv[2]#"../data/game_codedata_test.dat"
    number = sys.argv[3]
    train_feature, train_label = load_data(train_file)
    test_feature, test_label = load_data(test_file)
    classifierArray = train(train_feature,train_label,int(number))
    prediction = test(test_feature,classifierArray)
    ave_pro = evaluate(prediction,test_label)
    print 'accuracy: %r %%' %(1-1.0*sum(prediction!=mat(test_label).T)/len(prediction))
    print 'average probability: %r' %(ave_pro)

