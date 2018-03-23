Title: 5. Classifying Claims - Reporting Results
Tags: reporting_results
Authors: Ben Hoyle
Summary: This post looks at our results and what we have learnt from the project.

# 5. Classifying Claims - Reporting Results

Here we will look at what we have learnt during our project to classify patent claims. The aim of doing this is to consolidate what we have learnt, and practice a methodology for presenting our results.

The post [here](https://machinelearningmastery.com/how-to-use-machine-learning-results/) suggests the following structure for reporting our results:
1. Context - Define and describe why we are undertaking the project;
2. Problem - Describe the problem we are looking to solve;
3. Solution - Describe the solution our results suggest;
4. Findings - Present our results;
5. Limitations - Present the limitations we uncovered; and
6. Conclusions - Revisit and summarise the context, problem and solution (or why, question and answer)

We will look at these one by one.

## 1. Context

As described in our first post, the project was undertaken to practice applying a methodical approach to real-world patent data. 

Within the world of patent law, we have sets of patent claims, which set out the scope of legal protection. These claims, as they are contained in a patent application, are assigned a patent classification, which is a hierarchical code that categorises the subject-matter of the claims. 

Patent specifications are published and made available for download in bulk. This provides large labelled datasets we can use to practice supervised machine learning. In particular, we can extract a main independent claim and an International Patent Classification (IPC) code as data for machine learning algorithms. To simplify things, we can start with the first letter of the IPC code, which assigns each patent application to one of eight classes.

## 2. Problem

The problem we are looking to solve is to develop a machine learning algorithm that, given the text of a patent claim, can predict the first letter of the IPC code, i.e. the first level of subject matter classification.

## 3. Solution

Our results indicate that a multi-layer perceptron or a Naive Bayes classifier would be the best at performing this task. 

However, from our results it does not seem possible to implement an accurate classifier for this problem. The best solutions return an accuracy of around 60%. While better than chance given the proportions of the data (~20%) this is still not necessarily high enough to confidently apply a classifier in a processing pipeline, e.g. to differentially process the text of the patent claim. 

It would, though, appear possible to provide probabilities for classes that could support human decision making.   

## 4. Findings

Our finding will be split into three sections:
1. Findings regarding the nature of the problem and the data;
2. Findings regarding suitable machine learning algorithms; and
3. Findings regarding specific algorithms.

### 4.1 Findings regarding the nature of the problem and the data

A general finding was that changes in algorithms, parameters, and model archectures only led to small improvements in accuracy (e.g. often along the lines of 1-2%). Generally, the best classification accuracy was fixed at 60%.



The project reinforced the finding that success in text classification tasks is more about the data pre-processing than the machine learning algorithms. 

For example, a

### 4.2 Findings regarding suitable machine learning algorithms

The results of our spot-checks were as follows:

| Classifer     | Average Accuracy (%)| Standard Deviation (%)  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


* A random choice in line with the label proportions returned an average classification accuracy of 20.14% (standard deviation 0.13%)
* A Naive Bayes classifier returned an average classification accuracy of 58.90% (standard deviation 1.12%)
* A k-Nearest Neighbours classifier with 8 clusters returned an average classification accuracy of 30.15% (standard deviation 2.74%)
* A stochastic gradient descent classifier returned an average classification accuracy of 56.79% (standard deviation 0.75%)
* A multilayer perceClassifier MLPClassifier has an average classification accuracy of 61.20 (0.54)
Classifier DecisionTreeClassifier has an average classification accuracy of 47.33 (0.52)
Classifier RandomForestClassifier has an average classification accuracy of 50.01 (1.17)
Classifier AdaBoostClassifier has an average classification accuracy of 34.34 (1.31)
Classifier SVC has an average classification accuracy of 62.52 (0.80)
