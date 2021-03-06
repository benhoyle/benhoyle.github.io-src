Title: 3. Classifying Claims - Spot-Checking Algorithms
Tags: spot_checking
Authors: Ben Hoyle
Summary: This post looks at how we can perform initial spot testing of machine learning algorithms.

# 3. Classifying Claims - Spot-Checking Algorithms

Now we have our claim data and Section labels in numeric form, we can start evaluating different machine learning algorithms to get a feel for what may or may not work. A good method to apply is described [here](https://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/).

At this stage we want to determine some baseline results and apply vanilla implementations of common algorithms. From this we can choose one or two algorithms to investigate further. 

## Loading Our Data

We can load our X and Y as saved in part 2.


```python
import pickle
with open("encoded_data.pkl", "rb") as f:
    print("Loading data")
    X, Y = pickle.load(f)
    print("{0} claims and {1} classifications loaded".format(len(X), len(Y)))
```

    Loading data
    11238 claims and 11238 classifications loaded


## Performance Measure

The primary performance measure we are going to evaluate is classification accuracy, e.g. a percentage of correctly predicted classification labels. 

In this project it is also useful to look at the confusion matrix. As there may be some overlap between classes, this may be visible in the confusion matrix.

## Cross Validation

We will be applying 5-fold cross validation. This will train the classifier on 80% of the data, and test on the remaining 20%, providing an average measure for classification accuracy across 5 repetitions with different samples in each split. (10-fold cross validation would be slightly better but this was taking slightly too long on my old desktop machine.)

scikit-learn has a [module](http://scikit-learn.org/stable/modules/cross_validation.html) for applying k-fold cross validation. When the cv parameter is set to 5 this will run 5-fold validation. This can also be used with keras as explained in section 7 [here](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/). 

We also want to set a random seed for reproducibility (so the same random numbers are generated on repetitions of the classification).

## Algorithm Selection

We will start with 5-10 different algorithms. We need algorithms adapted for supervised multi-class classification. Luckily scikit-learn provides a [handy list](http://scikit-learn.org/stable/modules/multiclass.html#multiclass).

For the present problem these will include:
- random selection (for use as a baseline);
- naive Bayes;
- logistic regression;
- k-Nearest Neighbour classifier with k=8;
- multi-layer perceptron;
- support vector machines;
- decision tree classifier; and
- random-forest classifier.

This [example](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) from scikit-learn may be useful.

## Applying the Algorithms


```python
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

classifiers = [
    DummyClassifier(random_state=7),
    MultinomialNB(),
    KNeighborsClassifier(n_neighbors=8),
    SGDClassifier(max_iter=5, tol=None),
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    SVC()
]
```


```python
# We can use the class name to work out each
classifiers[0].__class__.__name__
```

    'DummyClassifier'

```python
results = list()
# Convert one hot to target integer
Y_integers = numpy.argmax(Y, axis=1)

for clf in classifiers:
    name = clf.__class__.__name__
    scores = cross_val_score(clf, X, Y_integers, cv=5)
    results.append((
        name, 
        scores.mean()*100, 
        scores.std()*100
    ))
        
    print(
        "Classifier {0} has an average classification accuracy of {1:.2f} ({2:.2f})".format(
            name, 
            scores.mean()*100, 
            scores.std()*100
        )    
    )
```

Results
```
Classifier DummyClassifier has an average classification accuracy of 20.14 (0.13)
Classifier MultinomialNB has an average classification accuracy of 58.90 (1.12)
Classifier KNeighborsClassifier has an average classification accuracy of 30.15 (2.74)
Classifier SGDClassifier has an average classification accuracy of 56.79 (0.75)
Classifier MLPClassifier has an average classification accuracy of 61.20 (0.54)
Classifier DecisionTreeClassifier has an average classification accuracy of 47.33 (0.52)
Classifier RandomForestClassifier has an average classification accuracy of 50.01 (1.17)
Classifier AdaBoostClassifier has an average classification accuracy of 34.34 (1.31)
Classifier SVC has an average classification accuracy of 62.52 (0.80)

```
Gaussian process led to a memory error so we left that one out.

## Observations

Our baseline - making a random choice based on the different proportions of labels available is around 20% accuracy, i.e. if we guess randomly we still correctly classify the claims 1 time out of 5. 

The k-Nearest Neighbours classifier does quite poorly. This indicates that clear groups within the TD-IDF vector space do not exist. The AdaBoost classifier has a comparative performance at around 35% accuracy. It would thus seem sensible to exclude these classifiers at this stage.

Tree-based classifiers - Decision Trees and Random Forest Classifiers - do a little better with around 50% accuracy. Random Forests appear to provide a small improvement (~ 3%) over Decision Trees. However, these are around 10% below the next set of classifiers, so again we will exclude these classifiers at this stage.

This leaves four classifiers with accuracies of around 60%: Naive Bayes (MultinomialNB), a Linear Support Vector Machine (SVM) trained with Stochastic Gradient Descent (SGDClassifier), an SVM with a radial basis function (RBF) kernel (SVC) and an Multi-Layer Perceptron (MLPClassifier). By comparing the linear SVM and MLP we can see that adding hidden layers to a linear model provides an improvement of around 4%. Also the SVC classifier scores highly, better than the linear SVM. It thus seems sensible to look at neural networks with hidden layers over linear regression models. The multinomial Naive Bayes classifier works surprisingly well, and slightly out-peforms the linear SVM. 

From these results it looks like deep-learning, i.e. multilayer neural networks could provide a good result. We will thus investigate these methods in more detail in the next post. It would also be worth varying some parameters of the SVC classifier to see if we can improve accuracy (although training time may be an issue).

As Bayesian methods come at things from a quite different perspective, it may also be worth investigating whether we can obtain any improvement over the Naive Bayes approach. 

### An Aside on Classification History

The results of this spot-check experiment rather nicely match the historic development of supervised classification algorithms. 

Before recent developments in deep-learning, linear classifiers such as linear SVMs offered the best performance. Bayesian methods then offered an alternative set of classifiers with a similar performance. Many early spam filters used Naive Bayes classifiers. 

The recent improvements in deep-learning offer the potential for gains over the previous state-of-the-art. However, as in our case, these gains are not huge with vanilla models (less than 5%). 
