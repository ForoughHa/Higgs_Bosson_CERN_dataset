# Introduction
 
This toolkit will provide a set of implemented methods for a binary classifier. You will have access to fully functioning implementations of all the following methods and can choose your prefered algorithm for the final classification task.
* Linear regression using gradient descent: _least_squares_
* Linear regression using stochastic gradient descent: _least_squares_GD_
* Least squares regression using normal equations: _least_squares_SGD_
* Logistic regression using gradient descent: _logistic_regression_
* Regularized logistic regression using gradient descent: _reg_logistic_regression_
* Ridge regression using normal equations: _ridge_regression_

## Dataset
The default dataset used for this project, are provided with the online competition based on one of the most
popular machine learning challenges recently: **_Finding the Higgs boson_**, using original data from CERN.

_Note_ that you can apply all the implemented methods on your own dataset for a binary classification.

# Application

## Data Preparation
As the first step of the process, the dataset must be prepared to be fed into different methods. This step includes:
* **Loading your dataset:** The data set must be in the form of a matrix _x_ with dimentions _N <a href="http://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\times" title="\times" /></a> D_ where _N_ corresponds to the number of samples and _D_ is the dimension of features for each sample. Moreover, the corresponding labels of the training data must be in the form of a _N <a href="http://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\times" title="\times" /></a> 1_ binray vector.
The function `load_csv('dataset.csv')` is implemented in the toolbox for this step. 


* **Cleaning your dataset:** If you choose to also clear your dataset of possoble outliers and invalid features, both training and test sets must go under feature proccessing steps. This procedure could include standardization, normalization, cleaning, and bounding your dataset.
In the provided toolkit, this step is performed and specifically adapted to the **_Finding the Higgs boson_** dataset. The `data_cleaning(dataset)`, `standardize_dataset(dataset)`, and `data_bounded(dataset)` are customized to serve this cause on our dataset.

Now that the data is ready, it can be passed into the prefered algorithm with rest of the required parameters.

## Evaluate Algorithm

The follosing steps construct the body of the _run.py_ script.

* **Load and prepare data:**

`train_data = load_csv('train.csv')`

Following this function, the unwanted rows and columns in the .csv file are omitted while the labels are also extracted from the main dataset.

* **Clean data:** 

`X = data_cleaning(X)`

Cleans the data set and replaces the unmeasured values (i.e. -999) with the mean value of that specific feature among properly measured samples.

`X = standardize_dataset(X)`

Standardizes the data by substracting the mean value and deviding by standard deviation with respect to different features.

`tx = data_bounded(X, times_SE = 3.0)`

Bounds the data and clears outliers.  _times_SE_ refers to the maximum multiple of the standard error larger than which the features must be omitted.

* **Initialize parameters:** 

According to the utilized method of classification, some parameters need to be defined by the user and passed to the function. These include the step size, the initial weight vector, regularization parameter, etc. All these parameters will be discussed in more detail in the corresponding algorithm.

* **Run algorithm:** 

In the last step, based on the prefered algorithm and defined parameters, the model is generated and the weight vector for the last iteration, the corresoinding loss function, and the mean accuracy score of the model on the validation data are reported. This is achieved in two simple steps:

**1. Cross Validation:** 

According to the specified number folds, a k-fold cross validation is performed to randomly devide the train set into slices of _train set_ and local _test set_. 

` train_idx, test_idx  = cross_validation_split(tx, n_folds)`

* Arguments: 

         dataset: The input data [numpy],
         n_folds: Number of required folds to be split into [int]
         
* Outputs: 

         train_index: Indices of the random section chosen as the train data in each step of learning [numpy array], 
         test_index: Indices of the random section chosen as the test data in each step of learning [numpy array]

Finaly the train matrix will be devided into _tx_train_ and _tx_test_ and their corresponding labels _y_train_ and _y_test_ in accordance to the extracted indices in each iteraion of k-fold cross validation


**2. Apply algorithm:**

After choosing the algorithm and corresponding parameters, the folowwong function will evaluate the coefficients and report the results.

`find_coeffs(y, tx, y_test, tx_test, algorithm, *args)`

* Arguments: 

         y: Corresponding labels of the train data [numpy array],
         tx: The random section chosen as the train data in each step of learning [numpy array], 
         y_test: Corresponding labels of the test data [numpy array],
         tx_test: The random section chosen as the test data in each step of learning [numpy array],        
         algorithm: Chosen algorithm for classification; It could be 'least_squares'/ 'least_squares_GD'/
         'least_squares_SGD'/'logistic_regression'/'reg_logistic_regression'/'ridge_regression' [name of function]
         *args: All other required parameters for the specified algorithm
         
* Outputs: 

         coef: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

## List of Implemented Methods:

**Least Square GD**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test, 'least_squares_GD', initial_w, max_iters, gamma)`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`least_squares_GD(y, tx, initial_w, max_iters, gamma)`

* Arguments: 

         tx: The random section chosen as the train data in each step of learning [numpy array], 
         y: The labels of randomly chosen train data [numpy array],
         max_iters: Maximum number of steps to run [int],
         gamma: Learning rate of gradient descent method [float]
        
* Outputs: 

         w: Classification coefficients [float],
         loss: Loss function calculated for the method [float]


**Least Square SGD**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test, 'least_squares_SGD', initial_w, max_iters, gamma)`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`least_squares_SGD(y, tx, initial_w, max_iters, gamma)`

* Arguments: 

         tx: The random section chosen as the train data in each step of learning [numpy array], 
         y: The labels of randomly chosen train data [numpy array],
         max_iters: Maximum number of steps to run [int],
         gamma: Learning rate of gradient descent method [float]
        
* Outputs: 

         w: Classification coefficients [float],
         loss: Loss function calculated for the method [float]


**Least Square Normal Equation**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test,'least_squares')`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`least_squares(y, tx)`

* Arguments: 

        tx: The random section chosen as the train data in each step of learning [numpy array], 
        y: Corresponding labels of the chosen data [numpy array],
        
* Outputs: 

        w: Classification coefficients [float],
        loss: Loss function calculated for the method [float]


**Ridge Regression**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test, 'ridge_regression', lambda_)`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`ridge_regression(y, tx, lambda_)`

* Arguments: 

        y: Corresponding labels of the chosen data [numpy array],
        tx: The random section chosen as the train data in each step of learning [numpy array], 
        lambda_ : Regularization parameter [float]
        
* Outputs: 

        w: Classification coefficients [float],
        loss: Loss function calculated for the method [float]  


**Logistic Regression**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test, 'logistic_regression', initial_w, max_iters, gamma)`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`logistic_regression (y, tx, initial_w, max_iters, gamma)`

* Arguments: 

         y: The labels of randomly chosen train data [numpy array],
         tx: The random section chosen as the train data in each step of learning [numpy array], 
         initial_w: Initial value for the weight vector [numpy array],
         max_iters: Maximum number of steps to run [int],
         gamma: Learning rate of gradient descent method) [float]
        
* Outputs: 

        w: Classification coefficients [float],
        loss: Loss function calculated for the method [float]



**Regularized Logistic Regression**

In the run code:

`coefficients, loss, accuracy = find_coeffs(y_train, tx_train, y_test, tx_test, 'reg_logistic_regression', lambda_, initial_w, max_iters, gamma)`

* Outputs: 

         coefficients: Classification coefficients [float],
         loss: Loss function calculated for the method [float],
         accuracy: Mean accuracy score of the used algorithm on test slice [float]
         

Underlying function:

`reg_logistic_regression (y, tx, lambda_, initial_w, max_iters, gamma)`

* Arguments: 

         y: The labels of randomly chosen train data [numpy array],
         tx: The random section chosen as the train data in each step of learning [numpy array], 
         lambda_ : Regularization parameter [float],
         initial_w: Initial value for the weight vector [numpy array],
         max_iters: Maximum number of steps to run [int],
         gamma: Learning rate of gradient descent method) [float]
        
* Outputs: 

        w: Classification coefficients [float],
        loss: Loss function calculated for the method [float]
        
        
        
## Inner Functions of The Toolkit

**Load .csv data to a numpy array:

`load_csv(filename)`

* Arguments: 

         filename: File name [string]

* Outputs: 

        dataset: Loaded dataset [numpy array]


**Normalize dataset columns to the range 0-1

_Note that this function was not utilized on our toolkit for the dataset in hand.

`normalize_dataset(dataset)`

* Arguments: 

         dataset: The input data [numpy]
         
 *Outputs: 
 
         dataset: Normalized dataset with all elements of each data point in range of (0,1) [numpy]


**Standardize dataset columns

`standardize_dataset(dataset)`

* Arguments: 

         dataset: The input data [numpy]
         
 *Outputs: 
 
         dataset: Standardize dataset [numpy]

**Calculate accuracy percentage using actual and predicted labels

`accuracy_metric(actual, predicted)`

* Arguments: 

         actual: Real labels of train data [int], 
         predicted: The implemented model's predicted labels [int]
         
* Outputs: 

         accuracy: Accuracy score in percentage [float] 


**Calculate yhat predictions and apply sigmoid function on predicted labels

`predict_sigmoid(data, coefficients)`

* Arguments: 

        data: Dataset to predict its labels [numpy array], 
        coefficients: Classification coefficients [float]
        
* Outputs: 

        yhat: Predicted labels [float]

**Bound outliers in dataset

`data_bounded(X_d, times_SE=3.0)`
  
* Arguments:

        X: Standardized data matrix to be bounded by 3 times of SE [Numpy array]
        
* Output:

        X_bounded: Bounded data matrix [Numpy Array]
        

**Clean undefined, unmeasured, or invalid data

`data_cleaning(X)`


* Arguments:

        X: Data matrix [Numpy array]
        
* Outputs:

        X_c: Cleaned data [Numpy array]
    


## Evaluate On Test Data

In the last step, we apply all the cleaning and preparation steps on our test set. Eventually after the classifier coefficients are computed using the specified algorithm, they will be applied on our test data and the output predicted labels will be imported to a .csv file using the function below:

` create_csv_submission(ids, y_pred, name)`

* Arguments: 

        ids: event ids associated with each prediction
         y_pred: predicted class labels 
         name: string name of .csv output file to be created
         
* Output:

       testLabels: File containing predicted test lables [.csv]
        


# Contributions

**Team Members:** _in alphabetical order_

**1. Farzaneh Habibollahi Saatlou**

**2. Forough Habibollahi Saatlou**

**3. Reza Hosseini Kouh Kamari**

All members contributed to the coding and report writing related tasks equaly.












