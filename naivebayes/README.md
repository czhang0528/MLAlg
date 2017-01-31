Description: 
+ Naive Bayes algorithm for multinomial models on UCI mushroom dataset</br>
+ The script is also suitable for multi-class classification with multiple feature dimensions</br>

Usage:
+ Install Python (https://www.python.org/) on your computer (assuming you have not already done so)</br>
  Python 2.7 is the preferred version because Python 3 is a totally different language than Python</br>
+ The format of training and testing data file is : (assume that you have prepared your training dataset and testing dataset)</br>
   label, feature_1, feature_2, feature_3 ...</br>
   .</br>
   .</br>
   .</br>
  Each line contains an instance and is ended by a '\n' character. </br>
  For classification, 'label' is any real number or string indicating the class label (multi-class is supported). </br>
  The 'feature_i' gives a feature (attribute) value.</br> 
  Labels in the testing file are only used to calculate accuracy or errors. If they are unknown, just fill the first column with any numbers.</br>
  The UCI mushroom dataset can be downloaded here : https://archive.ics.uci.edu/ml/datasets/Mushroom</br>
+ Run: python naive_bayes.py path_of_training_dataset path_of_testing_dataset path_of_predict_label</br>
  In which:
  path_of_training_dataset : string, is the whole path of training dataset (e.g. agaricus-lepiota.data.train.txt)</br>
  path_of_testing_dataset : string, is the whole path of testing dataset (e.g. agaricus-lepiota.data.test1.txt)</br>
  path_of_predict_label : string, is the whole path of output predicted label, one instance one line (e.g. output.txt)</br>
+ If you have any questions about running this script, feel free to contact author

