Author: Cheng Zhang

Contact: zhang DOT 7804 AT osu DOT edu / (614)264-0299

Date: 2/12/2017

Requirement：Install Python (https://www.python.org/), numpy and matplotlib on your computer (assuming you have not already done so). Note, Python 2.7 is the preferred version because Python 3 is a totally different language than Python

Description: 
+ K-means algorithm for homework #2
+ The home directory should be ./code/
+ Please refer to report.pdf for more analyses and experimental results

Step 1: 
+ Run: python kmeans_step_1.py <path_of_training_dataset>
+ e.g.: python kmeans_step_1.py hw2_training.txt

Step 2:
+ Run: python kmeans_step_2.py <path_of_training_dataset> <path_of_testing_dataset>
+ e.g.: python kmeans_step_2.py hw2_training.txt hw2_testing.txt

Step 3:
+ Run: python kmeans_step_3.py <path_of_training_dataset> <path_of_testing_dataset> <path_of_error_rate_out_file>
+ e.g.: python kmeans_step_3.py hw2_training.txt hw2_testing.txt result_stp3.out

Step 4:
+ Run: python kmeans_step_4.py <path_of_training_dataset> <path_of_testing_dataset> <path_of_error_rate_out_file>
+ e.g.: python kmeans_step_4.py hw2_training.txt hw2_testing.txt result_stp4.out

Bouns 1:
+ Run: python kmeans_b1.py <path_of_training_dataset> <path_of_testing_dataset> <path_of_error_rate_out_file>
+ e.g.: python kmeans_b1.py hw2_training_3d.txt hw2_testing_3d.txt result_b1_3d.out
        python kmeans_b1.py hw2_training_4d.txt hw2_testing_4d.txt result_b1_4d.out
        python kmeans_b1.py hw2_training_5d.txt hw2_testing_5d.txt result_b1_5d.out

Bouns 2:
+ Run: python kmeans_b2.py <path_of_training_dataset> <path_of_testing_dataset> <path_of_error_rate_out_file>
+ e.g.: python kmeans_b2.py hw2_training_w3.6.txt hw2_testing_w3.6.txt result_b2_w3.6.out
        python kmeans_b2.py hw2_training_w4.0.txt hw2_testing_w4.0.txt result_b2_w4.0.out
        python kmeans_b2.py hw2_training_w5.0.txt hw2_testing_w5.0.txt result_b2_w5.0.out

Bouns 3:
