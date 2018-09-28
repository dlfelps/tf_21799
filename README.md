# tf_21799
code to test https://github.com/tensorflow/tensorflow/issues/21799
(Issue with sparse labels)

iris_estimator.py : premade estimator

iris_keras.py : keras

iris_keras2estimator.py: keras -> estimator


## ERRORS: None (however, see below)

## RESULTS:
Accuracy (estimator): 0.966667

Accuracy (keras): 0.933333

Accuracy (keras -> estimator): 0.333333


Although the (keras -> estimator) code compiles and runs successfully, it only yields chance levels of accuracy (e.g. 3 species of iris ~ 0.33). I have also tested this for different datasets with 100 classes and the accuracy is ~ 0.01. 