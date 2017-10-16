What is Linear Regression 
Linear regression is the simplest and most widely used statistical technique for predictive modeling. It basically gives us an equation, where we have our features as independent variables, on which our target variable is dependent upon.

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor)

This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables. For example, relationship between rash driving and number of road accidents by a driver is best studied through regression.

What is the outcome from Linear Regression?
Linear Regression produces continuous outcome

What is equation of Linear Regression?
1)Single feature
y = b + mx
2)Multiple features
y = b + m0x0 + m1x1 + ...


y  - >dependent variable
x  -> are the independent variables 
m(thetas) -> are the coefficients. Coefficients are basically the weights assigned to the features, based on their importance.

What is the goal of Linear Regression ?

The goal of linear regression is to fit a line to a set of points. Consider the following data.

Let’s suppose we want to model the above set of points with a line. To do this we’ll use the standard y = mx + b line equation where m is the line’s slope and b is the line’s y-intercept.

To find the best line for our data, we need to find the best set of slope m and y-intercept b values.



What is the purpose of best of fit line?

The main purpose of the best fit line is that our predicted values should be closer to our actual or the observed values.

What is residuals?

we tend to minimize the difference between the values predicted by us and the observed values, and which is actually termed as error and these errors are also called as residuals.

What is sum of square error ?

https://www.google.co.in/search?rlz=1C1GGRV_enIN764&biw=1366&bih=662&tbm=isch&sa=1&q=sum+of+squares+error&oq=sum+of+squares+ero&gs_l=psy-ab.3.0.0i13k1l2j0i8i13i30k1.1915.2410.0.3203.4.4.0.0.0.0.120.336.0j3.3.0....0...1.1.64.psy-ab..1.3.336...0j0i24k1.0.h-2wTBxWuH0#imgrc=BwbIZupKUrl2kM:

How to compute sum of square error ?
https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png

 minimum sum of squared errors

How do we find best fit line?
There two common algorithms to find the right coefficients for minimum sum of squared errors to find best fit line.

1)Gradient Descent
2)Ordinary Least Sqaure 

what is Gradient Descent ?

Optimization technique for  Linear Regression 

Gradient Descent finds best parameters (m1) and (m2) for our learning algorithm.

What Gradient ?
is slope or direction.

How Gradient will help in minimising error

It help should be go up or down to get to minimum error by giving direction or slope.

https://www.google.co.in/search?q=gradient+descent&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjZ_J-wq8LWAhUFs48KHYWRALwQ_AUICygC&biw=1366&bih=662#imgrc=ZX6ENysK6oEhwM:


Gradient is like a bowl .we are trying to find the optimal value.

To calculate Gradient we are going to compute partial derivate s with respect our values ..they are b and m.

Gradient descent is an algorithm that minimizes cost functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the cost function.
This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.

To run gradient descent on  error function, we first need to compute its gradient


Positive  gradient
https://www.google.co.in/search?q=gradient+meaning&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiItvjktMLWAhXDQI8KHb5dB0cQ_AUICigB&biw=1366&bih=662#imgrc=4Pk5nEtEItOx8M:


Negative  gradient
https://www.google.co.in/search?q=gradient+meaning&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiItvjktMLWAhXDQI8KHb5dB0cQ_AUICigB&biw=1366&bih=662#imgrc=69F7rzVfxHdgVM:


We can initialize our search to start at any pair of m and b values (i.e., any line) and let the gradient descent algorithm march downhill on our error function towards the best line. Each iteration will update m and b to a line that yields slightly lower error than the previous iteration. The direction to move in for each iteration is calculated using the two partial derivatives

The direction to move in for each iteration is calculated using the two partial derivatives



def stepGradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        b_gradient += -(2/N) * (points[i].y - ((m_current*points[i].x) + b_current))
        m_gradient += -(2/N) * points[i].x * (points[i].y - ((m_current * points[i].x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

What is learningRate ?

The learningRate variable controls how large of a step we take downhill during each iteration. If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.




what are all assumptions in Regression?

There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables.

There should be no correlation between the residual (error) terms. Absence of this phenomenon is known as Autocorrelation.

The independent variables should not be correlated. Absence of this phenomenon is known as multicollinearity.

The error terms must have constant variance. This phenomenon is known as homoskedasticity. The presence of non-constant variance is referred to heteroskedasticity.


The error terms must be normally distributed.


What if these assumptions get violated ?


Let’s dive into specific assumptions and learn about their outcomes (if violated):

1. Linear and Additive:  If you fit a linear model to a non-linear, non-additive data set, the regression algorithm would fail to capture the trend mathematically, thus resulting in an inefficient model. Also, this will result in erroneous predictions on an unseen data set.

How to check: Look for residual vs fitted value plots (explained below). Also, you can include polynomial terms (X, X², X³) in your model to capture the non-linear effect.


What is R-squared?
We can evaluate the model performance using the metric R-square. 


R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. ... 100% indicates that the model explains all the variability of the response data around its mean.


How to find R-squared?
https://www.analyticsvidhya.com/wp-content/uploads/2015/10/R2.png


R-squared is always between 0 and 1:

0 indicates that the model explains NIL variability in the response data around its mean.
1 indicates that the model explains full variability in the response data around its mean.

What is uni-Variate Regression?
deal with single independent variables

What is Multi-Variate Regression?

deal with multiple independent variables


Important Points:

There must be linear relationship between independent and dependent variables
Multiple regression suffers from multicollinearity, autocorrelation, heteroskedasticity.

Linear Regression is very sensitive to Outliers. It can terribly affect the regression line and eventually the forecasted values.

Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable

In case of multiple independent variables, we can go with forward selection, backward elimination and step wise approach for selection of most significant independent variables.
 

What Ridge Regression?

Ridge Regression is a technique used when the data suffers from multicollinearity ( independent variables are highly correlated). In multicollinearity, even though the least squares estimates (OLS) are unbiased, their variances are large which deviates the observed value far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

This equation also has an error term. The complete equation becomes:


y=a+b*x+e (error term),  [error term is the value needed to correct for a prediction error between the observed and predicted value]

Ridge regression solves the multicollinearity problem through shrinkage parameter λ (lambda). 

What is l2 regularization?
Ridge Regression is also called as l2 regularization


what is Lasso Regression?

Lasso regression differs from ridge regression in a way that it uses absolute values in the penalty function, instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates) values which causes some of the parameter estimates to turn out exactly zero. Larger the penalty applied, further the estimates get shrunk towards absolute zero. This results to variable selection out of given n variables.

important Points:

The assumptions of this regression is same as least squared regression except normality is not to be assumed
It shrinks coefficients to zero (exactly zero), which certainly helps in feature selection
This is a regularization method and uses l1 regularization

What ElasticNet Regression?
ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained with L1 and L2 prior as regularizer. Elastic-net is useful when there are multiple features which are correlated. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

----------------------

What is Logistic Regression ?

Logistic Regression :produces discrete output

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. It produces discrete output.

Logistic Regression is likely the most commonly used algorithm for solving all classification problems.
Logistic Regression is part of a larger class of algorithms known as Generalized Linear Model (glm).


What is  Inferential Statistics?
Making inferences about the population from the sample.

Sampling Distribution helps to estimate the population statistic.


What is Confidence Interval?
confidence interval is a type of interval estimate from the sampling distribution which gives a range of values in which the population statistic may lie.

What T-tests

T-tests are very much similar to the z-scores, the only difference being that instead of the Population Standard Deviation, we now use the Sample Standard Deviation. The rest is same as before, calculating probabilities on basis of t-values.


Different types of t-tests?
1-sample t-test
Paired t-test
2-sample t-test

What is ANOVA?

ANOVA (Analysis of Variance) is used to check if at least one of two or more groups have statistically different means. Now, the question arises – Why do we need another test for checking the difference of means between independent groups? Why can we not use multiple t-tests to check for the difference in means?

The answer is simple. Multiple t-tests will have a compound effect on the error rate of the result. Performing t-test thrice will give an error rate of ~15% which is too high, whereas ANOVA keeps it at 5% for a 95% confidence interval.

To perform an ANOVA, you must have a continuous response variable and at least one categorical factor with two or more levels. ANOVA requires data from approximately normally distributed populations with equal variances between factor levels. However,

ANOVA is measured using a statistic known as F-Ratio. It is defined as the ratio of Mean Square (between groups) to the Mean Square (within group).

Mean Square (between groups) = Sum of Squares (between groups) / degree of freedom (between groups)

Mean Square (within group) = Sum of Squares (within group) / degree of freedom (within group)


https://fhssrsc.byu.edu/SitePages/ANOVA,%20t-tests,%20Regression,%20and%20Chi%20Square.aspx

what Chi-Sqaure?

Chi-Sqaure test is based on the proportions of the two or more groups. Simply it deals with categorical variables (Nominal Scale). Eg. Association between Smoking (Yes/No) vs. Drinking Cofee (Yes/No)
what Anova ?
Anova is based on the means of more than two groups (use t-test when there are two groups). It deals with quantitative dependent variable  (Interval Scale) with a categorical grouping variable to compare. Assumption of normality is checked often. Eg. Whether the average income of Managerial, Technical and Administrative staffs differs or the same.
Anova is example of linear models


GLM ...

What are 
exponential distribution, the Poisson distribution, gamma distribution, binomial or multinomial distributions

GLM can only handle between-subjects factors

what is multinomial 
when we have multiple categories of responses.

What  ordinal logistic regression?
Ordinal because we have an ordinal response variable

Imp points about logistic regression?

GLM does not assume a linear relationship between dependent and independent variables. However, it assumes a linear relationship between link function and independent variables in logit model.

what is ROC Curve?

 which methods is used for best fit the data in Logistic Regression?

Maximum Likelihood

Logistic regression uses maximum likely hood estimate for training a logistic regression.



Which of the following algorithms do we use for Variable Selection?

A) LASSO
B) Ridge
C) Both
D) None of these

Solution: A

In case of lasso we apply a absolute penality, after increasing the penality in lasso some of the coefficient of variables may become zero.

What is maximum likelihood estimation (MLE)

In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations given the parameters.


1. What is a Decision Tree ? How does it work ?

Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.

Example:-

Let’s say we have a sample of 30 students with three variables Gender (Boy/ Girl), Class( IX/ X) and Height (5 to 6 ft). 15 out of these 30 play cricket in leisure time. Now, I want to create a model to predict who will play cricket during leisure period? In this problem, we need to segregate students who play cricket in their leisure time based on highly significant input variable among all three.





This is where decision tree helps, it will segregate the students based on all values of three variable and identify the variable, which creates the best homogeneous sets of students (which are heterogeneous to each other). In the snapshot below, you can see that variable Gender is able to identify best homogeneous sets compared to the other two variables.



Important Terminology related to Decision Trees





Root Node: It represents entire population or sample and this further gets divided into two or more homogeneous sets.
Splitting: It is a process of dividing a node into two or more sub-nodes.
Decision Node: When a sub-node splits into further sub-nodes, then it is called decision node.
Leaf/ Terminal Node: Nodes do not split is called Leaf or Terminal node.

Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
Branch / Sub-Tree: A sub section of entire tree is called branch or sub-tree.
Parent and Child Node: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.


How does a tree decide where to split?

Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable. Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

The algorithm selection is also based on type of target variables. Let’s look at the four most commonly used algorithms in decision tree:


Gini index says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.

It works with categorical target variable “Success” or “Failure”.
It performs only Binary splits
Higher the value of Gini higher the homogeneity.
CART (Classification and Regression Tree) uses Gini method to create binary splits.
Steps to Calculate Gini for a split

Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure (p^2+q^2).
Calculate Gini for split using weighted Gini score of each node of that split


Split on Gender:

Calculate, Gini for sub-node Female = (0.2)*(0.2)+(0.8)*(0.8)=0.68
Gini for sub-node Male = (0.65)*(0.65)+(0.35)*(0.35)=0.55
Calculate weighted Gini for Split Gender = (10/30)*0.68+(20/30)*0.55 = 0.59
Similar for Split on Class:

Gini for sub-node Class IX = (0.43)*(0.43)+(0.57)*(0.57)=0.51
Gini for sub-node Class X = (0.56)*(0.56)+(0.44)*(0.44)=0.51
Calculate weighted Gini for Split Class = (14/30)*0.51+(16/30)*0.51 = 0.51
Above, you can see that Gini score for Split on Gender is higher than Split on Class, hence, the node split will take place on Gender.


Chi-Square

It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.



 Are tree based models better than linear models?

 If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!

What are ensemble methods in tree based modeling ?

The literary meaning of word ‘ensemble’ is group. Ensemble methods involve group of predictive models to achieve a better accuracy and model stability. Ensemble methods are known to impart supreme boost to tree based models.

What is Bagging? How does it work?

Bagging is a technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set.

The steps followed in bagging are:

Create Multiple DataSets:
Sampling is done with replacement on the original data and new datasets are formed.
The new data sets can have a fraction of the columns as well as rows, which are generally hyper-parameters in a bagging model
Taking row and column fractions less than 1 helps in making robust models, less prone to overfitting
Build Multiple Classifiers:
Classifiers are built on each data set.
Generally the same classifier is modeled on each data set and predictions are made.
Combine Classifiers:
The predictions of all the classifiers are combined using a mean, median or mode value depending on the problem at hand.
The combined values are generally more robust than a single model.

Ensemble learning ?
Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem.Dec 22, 2008

Random Forest is a bagging algorithm rather than a boosting algorithm. They are two opposite way to achieve a low error.


Bagging VS Boosting
parallel ensemble: each model is built independently
aim to decrease variance, not bias
suitable for high variance low bias models (complex models)
an example of a tree based method is random forest, which develop fully grown trees (note that RF modifies the grown procedure to reduce the correlation between trees)
Boosting:

sequential ensemble: try to add new models that do well where previous models lack
aim to decrease bias, not variance
suitable for low variance high bias models
an example of a tree based method is gradient boosting




What is a Support Vector and what is SVM?

Support Vectors are simply the co-ordinates of individual observation. For instance, (45,150) is a support vector which corresponds to a female.

Support Vector Machine is a line which best segregates the Male from the Females. In this case, the two classes are well separated from each other, hence it is easier to find a SVM.

The easiest way to interpret the objective function in a SVM is to find the minimum distance of the frontier from closest support vector


Once we have these distances for all the frontiers, we simply choose the frontier with the maximum distance (from the closest support vector). 

What if we do not find a clean frontier which segregates the classes?


we need to map these vector to a higher dimension plane so that they get segregated from each other. Such cases will be covered once we start with the formulation of SVM. For now,
























http://www.listendata.com/2015/09/linear-regression-with-r.html







