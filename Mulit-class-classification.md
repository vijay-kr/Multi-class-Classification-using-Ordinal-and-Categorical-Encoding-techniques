
# Illustration of Multi Label Classification on the cars dataset

### Importing all the required Libraries and statements to avoid warning


```python
%matplotlib inline
from sklearn.datasets import load_breast_cancer
from sklearn import tree,linear_model,neighbors, datasets
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import label_binarize, StandardScaler
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
# Ignoring warnings for clean output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
```

### Loading the cars dataset and exploring the dataset to understand the variables and target variable


```python
cars = pd.read_csv('car.data',header = None)
```


```python
cars.head(3)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
  </tbody>
</table>





```python
cars.columns = ['buying','maint','doors',
                     'persons','lug_boot','safety','class']
```


```python
cars.shape
```


    (1728, 7)

the dataset has around 1728 records with 7 variable, the one 6 is the features and the 7 variable is the target variable (class) which needs to be classified 


```python
cars.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
      <td>1728</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>high</td>
      <td>high</td>
      <td>5more</td>
      <td>more</td>
      <td>big</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>432</td>
      <td>432</td>
      <td>432</td>
      <td>576</td>
      <td>576</td>
      <td>576</td>
      <td>1210</td>
    </tr>
  </tbody>
</table>


It can be observed that all the features have all the values so missing data treatment is not required, all the features are ordinal since they have 4 distinct classes which have order.

Splitting the target variable from features so that the features can be pre processed before training


```python
features = cars.loc[:,'buying':'safety']
features1 = features
target = cars[['class']]
```


```python
features.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
    </tr>
  </tbody>
</table>



```python
features_one_hot = pd.get_dummies(features1, drop_first=True)
```


```python
features_one_hot.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying_low</th>
      <th>buying_med</th>
      <th>buying_vhigh</th>
      <th>maint_low</th>
      <th>maint_med</th>
      <th>maint_vhigh</th>
      <th>doors_3</th>
      <th>doors_4</th>
      <th>doors_5more</th>
      <th>persons_4</th>
      <th>persons_more</th>
      <th>lug_boot_med</th>
      <th>lug_boot_small</th>
      <th>safety_low</th>
      <th>safety_med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



```python
target['class'].unique()
```


    array(['unacc', 'acc', 'vgood', 'good'], dtype=object)


```python
classes = {'unacc': 0,'acc': 1,'good':2,'vgood':3} 
 
target['class'] = [classes[item] for item in target['class']]
```

```python
target['class'].value_counts()
```


    0    1210
    1     384
    2      69
    3      65
    Name: class, dtype: int64


```python
enc = OrdinalEncoder()
features = enc.fit_transform(features)
features = pd.DataFrame(features)
```


```python
features.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>



```python
features.columns = ['buying','maint','doors',
                     'persons','lug_boot','safety']
```


```python
features.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buying</th>
      <th>maint</th>
      <th>doors</th>
      <th>persons</th>
      <th>lug_boot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
Now that entire data is converted to numerical, we shall start with the modelling process 

### Splitting data in train and test 


```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30,random_state=45,stratify = target)
X1_train, X1_test, y1_train, y1_test = train_test_split(features_one_hot, target, test_size=0.30,random_state=44,stratify = target)
```


```python
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))
```


    {0: 847, 1: 269, 2: 48, 3: 45}


```python
unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))
```


    {0: 363, 1: 115, 2: 21, 3: 20}

the data is split in the required ratio and the class labels ratio is maintained hence, there is no need for stratified sampled. we can continue with random split

# Decision Tree

Algorithms like decision tree has many hyperparameters which we can tweak. Grid Search method from sklearn can be used so that we can test a lot of hyperparameters and do cross validation of each to get the best set of hyper parameters.
Below is the code for decision tree hyperparameter optimization using grid search

Max_depth, min_samples in leaf and min_impurity_decrease hyper parameters is used for the model because max depth and min samples in leaf nodes should put a constraint on the tree growing full to the each individual node which would lead to overfitting and min_impurity_decrease is used to deal with underfitting because if a node is impurity with this threshold the try will try to split it and try to make the leaf nodes pure than the parent.


```python
tuned_parameters = {'max_depth': np.arange(3,7),'min_samples_leaf': np.arange(5,30),"criterion":["gini","entropy"],"min_impurity_decrease":[1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1]}

inner_cv = KFold(n_splits=4, shuffle=True)
outer_cv = KFold(n_splits=4, shuffle=True)

grid_tree = tree.DecisionTreeClassifier(random_state=45)

#Nested CV inner loop
grid = GridSearchCV(grid_tree, tuned_parameters, cv = inner_cv, scoring='accuracy')
grid.fit(X_train,y_train)

#Nested CV outer loop
nested_score = cross_val_score(grid, features, target, cv=outer_cv,scoring ='accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid.best_estimator_)
y_pred = grid.best_estimator_.predict(X_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y_test, y_pred))
print()
print("Classification Report: - \n",classification_report(y_test, y_pred))
```

    Using Nested CV with grid search,accuracy: 87.33% +/- 3.30%
    
    The best hyper-parameters to get this accuracy is :-
     {'criterion': 'entropy', 'max_depth': 6, 'min_impurity_decrease': 1e-07, 'min_samples_leaf': 7}
    
    The best decision tree classifier is :-
     DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=6,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=1e-07, min_impurity_split=None,
                min_samples_leaf=7, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=45,
                splitter='best')
    Confusion Matrix: - 
     [[327  33   3   0]
     [  0  97  11   7]
     [  0  10   9   2]
     [  0   0   0  20]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       1.00      0.90      0.95       363
               1       0.69      0.84      0.76       115
               2       0.39      0.43      0.41        21
               3       0.69      1.00      0.82        20
    
       micro avg       0.87      0.87      0.87       519
       macro avg       0.69      0.79      0.73       519
    weighted avg       0.90      0.87      0.88       519    

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 87.33% +/- 3.30%** with a **precision of 90% and recall of 87%**

Our model was chosen based of **f1-score which is 88%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.


```python
tuned_parameters = {'max_depth': np.arange(3,7),'min_samples_leaf': np.arange(5,30),"criterion":["gini","entropy"],"min_impurity_decrease":[1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1]}

inner_cv = KFold(n_splits=4, shuffle=True)
outer_cv = KFold(n_splits=4, shuffle=True)

grid_tree = tree.DecisionTreeClassifier(random_state=44)

#Nested CV inner loop
grid = GridSearchCV(grid_tree, tuned_parameters, cv = inner_cv, scoring='accuracy')
grid.fit(X1_train,y1_train)

#Nested CV outer loop
nested_score = cross_val_score(grid, features_one_hot, target, cv=outer_cv,scoring ='accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid.best_estimator_)
y1_pred = grid.best_estimator_.predict(X1_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y1_test, y1_pred))
print()
print("Classification Report: - \n",classification_report(y1_test, y1_pred))
```

    Using Nested CV with grid search,accuracy: 83.62% +/- 2.13%
    
    The best hyper-parameters to get this accuracy is :-
     {'criterion': 'gini', 'max_depth': 6, 'min_impurity_decrease': 1e-07, 'min_samples_leaf': 6}
    
    The best decision tree classifier is :-
     DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=1e-07, min_impurity_split=None,
                min_samples_leaf=6, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=44,
                splitter='best')
    Confusion Matrix: - 
     [[328  35   0   0]
     [ 13  91  11   0]
     [  0  12   6   3]
     [  0  13   3   4]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.96      0.90      0.93       363
               1       0.60      0.79      0.68       115
               2       0.30      0.29      0.29        21
               3       0.57      0.20      0.30        20
    
       micro avg       0.83      0.83      0.83       519
       macro avg       0.61      0.55      0.55       519
    weighted avg       0.84      0.83      0.83       519   

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 83.62% +/- 2.13%** with a **precision of 84% and recall of 83%**

Our model was chosen based of **f1-score which is 83%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.

### 2. K-NN  

KNN is based on distances between data points, since we have ordinal variables we cannot say that difference between 1-2 is **not same** as 2-3 so for KNN we are running the one hot encoded version.


```python
param_grid = {'n_neighbors' : np.arange(1,30), 'weights' : ['uniform','distance']}

grid_knn_clf = neighbors.KNeighborsClassifier()

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

#Nested CV innner loop
grid_knn = GridSearchCV(grid_knn_clf, param_grid, cv = inner_cv, scoring='accuracy')
grid_knn.fit(X1_train,y1_train)

#Nested CV outer loop
nested_score = cross_val_score(grid_knn, features_one_hot, target, cv=outer_cv,scoring='accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_knn.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_knn.best_estimator_)
y1_pred = grid_knn.best_estimator_.predict(X1_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y1_test, y1_pred))
print()
print("Classification Report: - \n",classification_report(y1_test, y1_pred))
```

    Using Nested CV with grid search,accuracy: 82.47% +/- 4.01%
    
    The best hyper-parameters to get this accuracy is :-
     {'n_neighbors': 10, 'weights': 'distance'}
    
    The best decision tree classifier is :-
     KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=10, p=2,
               weights='distance')
    Confusion Matrix: - 
     [[355   8   0   0]
     [ 39  74   2   0]
     [  9   8   3   1]
     [  7   5   2   6]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.87      0.98      0.92       363
               1       0.78      0.64      0.70       115
               2       0.43      0.14      0.21        21
               3       0.86      0.30      0.44        20
    
       micro avg       0.84      0.84      0.84       519
       macro avg       0.73      0.52      0.57       519
    weighted avg       0.83      0.84      0.82       519   

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 82.47% +/- 4.01%** with a **precision of 83% and recall of 84%**

Our model was chosen based of **f1-score which is 82%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.

## 3. Logistic Regression


```python
grid_values = {
               'C':[1e-4,0.001,.009,0.01,.09,1,5,10,25,100,1000,1e4],
               'multi_class' : ['multinomial'],
              'solver': ['lbfgs']}

grid_log_clf = linear_model.LogisticRegression(random_state=45)

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_logit = GridSearchCV(grid_log_clf, grid_values, cv = inner_cv, scoring='accuracy')
grid_logit.fit(X_train,y_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_logit, features, target, cv=outer_cv,scoring = 'accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_logit.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_logit.best_estimator_)
y_pred = grid_logit.best_estimator_.predict(X_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y_test, y_pred))
print()
print("Classification Report: - \n",classification_report(y_test, y_pred))
```

    Using Nested CV with grid search,accuracy: 70.14% +/- 4.61%
    
    The best hyper-parameters to get this accuracy is :-
     {'C': 0.09, 'multi_class': 'multinomial', 'solver': 'lbfgs'}
    
    The best decision tree classifier is :-
     LogisticRegression(C=0.09, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='multinomial',
              n_jobs=None, penalty='l2', random_state=45, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False)
    Confusion Matrix: - 
     [[336  27   0   0]
     [ 91  24   0   0]
     [ 19   2   0   0]
     [ 14   6   0   0]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.73      0.93      0.82       363
               1       0.41      0.21      0.28       115
               2       0.00      0.00      0.00        21
               3       0.00      0.00      0.00        20
    
       micro avg       0.69      0.69      0.69       519
       macro avg       0.28      0.28      0.27       519
    weighted avg       0.60      0.69      0.63       519


**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 70.14% +/- 4.61%** with a **precision of 60% and recall of 69%**

Our model was chosen based of **f1-score which is 63%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.


```python
grid_values = {'penalty': ['l1', 'l2'], \
               'C':[1e-4,0.001,.009,0.01,.09,1,5,10,25,100,1000,1e4],
               'multi_class' : ['multinomial'],
              'solver': ['saga']}

grid_log_clf = linear_model.LogisticRegression(random_state=44)

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_logit = GridSearchCV(grid_log_clf, grid_values, cv = inner_cv, scoring='accuracy')
grid_logit.fit(X1_train,y1_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_logit, features_one_hot, target, cv=outer_cv,scoring = 'accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_logit.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_logit.best_estimator_)
y1_pred = grid_logit.best_estimator_.predict(X1_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y1_test, y1_pred))
print()
print("Classification Report: - \n",classification_report(y1_test, y1_pred))
```

    Using Nested CV with grid search,accuracy: 92.82% +/- 2.20%
    
    The best hyper-parameters to get this accuracy is :-
     {'C': 100, 'multi_class': 'multinomial', 'penalty': 'l1', 'solver': 'saga'}
    
    The best decision tree classifier is :-
     LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='multinomial',
              n_jobs=None, penalty='l1', random_state=44, solver='saga',
              tol=0.0001, verbose=0, warm_start=False)
    Confusion Matrix: - 
     [[353   9   1   0]
     [ 21  91   1   2]
     [  0   2  14   5]
     [  0   0   1  19]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.94      0.97      0.96       363
               1       0.89      0.79      0.84       115
               2       0.82      0.67      0.74        21
               3       0.73      0.95      0.83        20
    
       micro avg       0.92      0.92      0.92       519
       macro avg       0.85      0.85      0.84       519
    weighted avg       0.92      0.92      0.92       519  

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 92.82% +/- 2.20%** with a **precision of 92% and recall of 92%**

Our model was chosen based of **f1-score which is 92%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.

## 4. Naive Bayesian


```python
grid_values = {'alpha' : [1,2,3,4,5,6,7,8,9,10]}

grid_NB_clf = MultinomialNB()

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_NB = GridSearchCV(grid_NB_clf, grid_values, cv = inner_cv, scoring='accuracy')
grid_NB.fit(X_train,y_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_NB, features, target, cv=outer_cv,scoring = 'accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_NB.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_NB.best_estimator_)
y_pred = grid_NB.best_estimator_.predict(X_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y_test, y_pred))
print()
print("Classification Report: - \n",classification_report(y_test, y_pred))
```

    Using Nested CV with grid search,accuracy: 70.08% +/- 4.55%
    
    The best hyper-parameters to get this accuracy is :-
     {'alpha': 1}
    
    The best decision tree classifier is :-
     MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
    Confusion Matrix: - 
     [[363   0   0   0]
     [113   2   0   0]
     [ 21   0   0   0]
     [ 20   0   0   0]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.70      1.00      0.82       363
               1       1.00      0.02      0.03       115
               2       0.00      0.00      0.00        21
               3       0.00      0.00      0.00        20
    
       micro avg       0.70      0.70      0.70       519
       macro avg       0.43      0.25      0.21       519
    weighted avg       0.71      0.70      0.58       519


**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 70.08% +/- 4.55%** with a **precision of 71% and recall of 70%**

Our model was chosen based of **f1-score which is 58%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.


```python
grid_values = {'alpha' : [1,2,3,4,5,6,7,8,9,10]}

grid_NB_clf = MultinomialNB()

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_NB = GridSearchCV(grid_NB_clf, grid_values, cv = inner_cv, scoring='accuracy')
grid_NB.fit(X1_train,y1_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_NB, features_one_hot, target, cv=outer_cv,scoring = 'accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_NB.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_NB.best_estimator_)
y1_pred = grid_NB.best_estimator_.predict(X1_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y1_test, y1_pred))
print()
print("Classification Report: - \n",classification_report(y1_test, y1_pred))
```

    Using Nested CV with grid search,accuracy: 73.78% +/- 5.51%
    
    The best hyper-parameters to get this accuracy is :-
     {'alpha': 1}
    
    The best decision tree classifier is :-
     MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
    Confusion Matrix: - 
     [[360   3   0   0]
     [ 87  28   0   0]
     [ 13   8   0   0]
     [ 16   4   0   0]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       0.76      0.99      0.86       363
               1       0.65      0.24      0.35       115
               2       0.00      0.00      0.00        21
               3       0.00      0.00      0.00        20
    
       micro avg       0.75      0.75      0.75       519
       macro avg       0.35      0.31      0.30       519
    weighted avg       0.67      0.75      0.68       519


**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 73.78% +/- 5.51%** with a **precision of 67% and recall of 75%**

Our model was chosen based of **f1-score which is 68%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.

## 5. SVM


```python
param_grid = {'kernel':['linear','rbf'],'C': [0.01, 0.1, 1, 10, 100], 'gamma' :[0.001, 0.01, 0.1, 1]}

grid_svc_clf= SVC(random_state = 45)

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_svm = GridSearchCV(grid_svc_clf, param_grid, cv = inner_cv, scoring='accuracy')
grid_svm.fit(X_train,y_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_svm, features, target, cv=outer_cv,scoring='accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_svm.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_svm.best_estimator_)
y_pred = grid_svm.best_estimator_.predict(X_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y_test, y_pred))
print()
print("Classification Report: - \n",classification_report(y_test, y_pred))
```

    Using Nested CV with grid search,accuracy: 99.25% +/- 0.76%
    
    The best hyper-parameters to get this accuracy is :-
     {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
    
    The best decision tree classifier is :-
     SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, probability=False, random_state=45, shrinking=True,
      tol=0.001, verbose=False)
    Confusion Matrix: - 
     [[360   3   0   0]
     [  0 115   0   0]
     [  0   0  21   0]
     [  0   1   0  19]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       1.00      0.99      1.00       363
               1       0.97      1.00      0.98       115
               2       1.00      1.00      1.00        21
               3       1.00      0.95      0.97        20
    
       micro avg       0.99      0.99      0.99       519
       macro avg       0.99      0.99      0.99       519
    weighted avg       0.99      0.99      0.99       519    

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 99.25% +/- 0.76%** with a **precision of 99% and recall of 99%**

Our model was chosen based of **f1-score which is 99%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.


```python
param_grid = {'kernel':['linear','rbf'],'C': [0.01, 0.1, 1, 10, 100], 'gamma' :[0.001, 0.01, 0.1, 1]}

grid_svc_clf= SVC(random_state = 45)

inner_cv = KFold(n_splits=4, shuffle=True, random_state=45)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=45)

grid_svm = GridSearchCV(grid_svc_clf, param_grid, cv = inner_cv, scoring='accuracy')
grid_svm.fit(X1_train,y1_train)

# Nested CV with parameter optimization
nested_score = cross_val_score(grid_svm, features_one_hot, target, cv=outer_cv,scoring='accuracy')
```


```python
# Mean Accuracy with +/- 2 std deviations
print("Using Nested CV with grid search,accuracy: {0:.2%} +/- {1:.2%}".format(nested_score.mean(), nested_score.std() * 2))
print()
print ("The best hyper-parameters to get this accuracy is :-\n", grid_svm.best_params_)
print()
print ("The best decision tree classifier is :-\n", grid_svm.best_estimator_)
y1_pred = grid_svm.best_estimator_.predict(X1_test)

#Goodness Measures confusion matrix and other measures like accuracy, precision,recall
print("Confusion Matrix: - \n",confusion_matrix(y1_test, y1_pred))
print()
print("Classification Report: - \n",classification_report(y1_test, y1_pred))
```

    Using Nested CV with grid search,accuracy: 99.25% +/- 1.15%
    
    The best hyper-parameters to get this accuracy is :-
     {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
    
    The best decision tree classifier is :-
     SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, probability=False, random_state=45, shrinking=True,
      tol=0.001, verbose=False)
    Confusion Matrix: - 
     [[362   1   0   0]
     [  1 114   0   0]
     [  0   0  21   0]
     [  0   0   0  20]]
    
    Classification Report: - 
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       363
               1       0.99      0.99      0.99       115
               2       1.00      1.00      1.00        21
               3       1.00      1.00      1.00        20
    
       micro avg       1.00      1.00      1.00       519
       macro avg       1.00      1.00      1.00       519
    weighted avg       1.00      1.00      1.00       519  

**Model Goodness**

The class distribution is 1210 unacceptable, 384 acceptable ,69 good, 65 very good cars, the decision tree classifier without one hot encoding gives an **Accuracy of 99.25% +/- 1.15%** with a **precision of 100% and recall of 100%**

Our model was chosen based of **f1-score which is 100%** which is the harmonic mean of precision and recall and hence a good measure to determine a good fit.

Of the 9 classifiers test above SVM gives the highest accuracy of 99.25 +/- 0.76%. 
 [[360   3   0   0]
 [  0 115   0   0]
 [  0   0  21   0]
 [  0   1   0  19]]

 Above is the confusion for the same. it can be seen that classes 1,3,4 are accurately predicted whereas only class 2 has some misclassifications, 3 instances of class 2 has been predicted as class 1 and 1 instance as class 4. this is fine because these small misclassifications could be because of outliers of class 2. Overall SVM is able to accurately classify all classes and is a very good classifier.

For the classifier like SVM, Naïve Bayes, Logistic and decision tree both the one hot encoded version as well as the numerical methods was run. It was noticed that one hot encoded (categorical) version gave a better or same accuracy as the numerical data especially for logistic regression it is seen that the accuracy increases from 70% to 92% when the data is changed to to categorical.

Pros of one hot encoding :
Since the values of a feature is represented as separate column, effect of individual value in classification can be used in modelling process

Cons of one hot encoding :
The number of dimensions increases which in turn may lead to poor model

Pros of Numerical :
Computation is faster when the data is made numerical as compared one hot encoding

Cons of numerical:
When the data is ordinal, the difference between 1-2 may not be the same as 2-3 so for algorithms like KNN it may not be helpful.


