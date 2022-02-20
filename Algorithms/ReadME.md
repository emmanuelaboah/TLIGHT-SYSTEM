## Algorithms
This directory contains OCSVM, OCNN, and IF algorithms described in our <i class="icon-cog"></i> **[work](https://)**. 

### OCSVM
This subdirectory contains ```tlight_ocsvm.py``` script.
The script is responsible for training and evaluating the OCSVM model on 
the train and test datasets. Hyper-parameters specified in the alg are the
same as the optimal hyper-parameters specified in our work.

**Note**: ```There are five different test sets in the datasets 
directory, hence the default name of the test data in the script 
needs to be changed each time the alg is evaluated on a different test set ```

### OCNN
This sub-directory contains ````tlight_ocnn.py```` script. The script contains
the algorithm for OCNN with all required hyper-parameters specified in our
work. The script trains the OCNN alg on the training dataset and evaluates the
model on the test dataset.

### IF
This sub-directory contains two scripts, namely:
1. **IF_hyper_tune.py**: This script takes a range of IF hyper-parameters and
trains the IF model to select the best hyperparameters.


2. **tlight_iforest.py**: This script is used to train and evaluate the IF
the algorithm on the train and test dataset using the best hyper-parameters.
