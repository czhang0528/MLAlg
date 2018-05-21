Author: Cheng Zhang

Contact: zhang DOT 7804 AT osu DOT edu / (614)264-0299

Date: 2/24/2017

Reference: https://github.com/jakezhaojb/adaboost-py

Requirement：
+ Install Python (https://www.python.org/) and numpy on your computer (assuming you have not already done so). 
+ Note, Python 2.7 is the preferred version because Python 3 is a totally different language than Python

Description: 
+ Adaboost algorithm for homework #3
+ The home directory should be ./code/

Usage:
+ Run: python adaboost.py <path_of_training_dataset> <path_of_training_dataset> <number_of_classifiers/interations>
+ e.g., for question (a), please run: python adaboost.py game_codedata_train.dat game_codedata_test.dat 1
+ Or you can change the number: python adaboost.py game_codedata_train.dat game_codedata_test.dat 2
								python adaboost.py game_codedata_train.dat game_codedata_test.dat 3
								.....
+ Output: accuracy, average probability, and the best decision stump at current status