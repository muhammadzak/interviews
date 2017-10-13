
Linear Regression 

Linear regression is the simplest and most widely used statistical technique for predictive modeling. It basically gives us an equation, where we have our features as independent variables, on which our target variable is dependent upon.

y = b + mx

#y = b + m0x0 + m1x1 + ...


y as our dependent variable

x is are the independent variables and all m(thetas) are the coefficients. Coefficients are basically the weights assigned to the features, based on their importance.


Linear Regression produces continuous outcome

Best of fit line

The main purpose of the best fit line is that our predicted values should be closer to our actual or the observed values,

we tend to minimize the difference between the values predicted by us and the observed values, and which is actually termed as error

These errors are also called as residuals


sum of square error ->

https://www.google.co.in/search?rlz=1C1GGRV_enIN764&biw=1366&bih=662&tbm=isch&sa=1&q=sum+of+squares+error&oq=sum+of+squares+ero&gs_l=psy-ab.3.0.0i13k1l2j0i8i13i30k1.1915.2410.0.3203.4.4.0.0.0.0.120.336.0j3.3.0....0...1.1.64.psy-ab..1.3.336...0j0i24k1.0.h-2wTBxWuH0#imgrc=BwbIZupKUrl2kM:

How to compute sum of square error
https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png

we have minimize rmse using gradient descent



Our main objective is to find out the error and minimize it.




Gradient Descent : optimization technique for  Linear Regression 
We have find best fit using Gradient Descent

We have to minimize the error using Gradient Descent

Gradient Descent finds best parameters (m1) and (m2) for our learning algorithm.

There are many types of gradient descent algorithms. They can be classified by two methods mainly

1)On the basis of data ingestion
Full Batch Gradient Descent Algorithm
Stochastic Gradient Descent Algorithm
In full batch gradient descent algorithms, you use whole data at once to compute the gradient, whereas in stochastic you take a sample while computing the gradient.


2)On the basis of differentiation techniques 
First order Differentiation
Second order Differentiation
Gradient descent requires calculation of gradient by differentiation of cost function. We can either use first order differentiation or second order differentiation.

Challenges in executing Gradient Descent

Data challenges

If the data is arranged in a way that it poses a non-convex optimization problem.(http://www.cs.ubc.ca/labs/lci/mlrg/slides/non_convex_optimization.pdf) It is very difficult to perform optimization using gradient descent. Gradient descent only works for problems which have a well defined convex optimization problem.
Even when optimizing a convex optimization problem, there may be numerous minimal points. The lowest point is called global minimum, whereas rest of the points are called local minima. Our aim is to go to global minimum while avoiding local minima.

Gradient Challenges

If the execution is not done properly while using gradient descent, it may lead to problems like vanishing gradient or exploding gradient problems. These problems occur when the gradient is too small or too large. And because of this problem the algorithms do not converge.


Implementation Challenges

Most of the neural network practitioners don’t generally pay attention to implementation, but it’s very important to look at the resource utilization by networks. For eg: When implementing gradient descent, it is very important to note how many resources you would require. If the memory is too small for your application, then the network would fail.
Also, its important to keep track of things like floating point considerations and hardware / software prerequisites.

Variants of Gradient Descent algorithms



Implementation challenges


learning rate :Hyper parameter ...how fast our model learning...
if learning rate is too low...model will take lot of time converge 
if learning rate is too high ...model may not converge 

y = mx+b
m ->slope
b ->Intercept 


number of iteration :Number of times you want to execute code




Gradient  is slope or direction.

Gradient will help should be go up or down to get to minimum error

https://www.google.co.in/search?q=gradient+descent&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjZ_J-wq8LWAhUFs48KHYWRALwQ_AUICygC&biw=1366&bih=662#imgrc=ZX6ENysK6oEhwM:

Gradient is like a bowl ...we are trying to find the optimal value

To calculate Gradient we are going to compute partial derivate s with respect our values ..they are b and m

...

When we complex function we will have local and global minima



https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/


Gradient descent is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function.
This iterative minimization is achieved using calculus, taking steps in the negative direction of the function gradient.


Linear Regression
The goal of linear regression is to fit a line to a set of points. Consider the following data.

Let’s suppose we want to model the above set of points with a line. To do this we’ll use the standard y = mx + b line equation where m is the line’s slope and b is the line’s y-intercept.

To find the best line for our data, we need to find the best set of slope m and y-intercept b values.


A standard approach to solving this type of problem is to define an error function (also called a cost function) that measures how “good” a given line is. 
This function will take in a (m,b) pair and return an error value based on how well the line fits our data. To compute this error for a given line, we’ll iterate through each (x,y) point in our data set and sum the square distances between each point’s y value and the candidate line’s y value (computed at mx + b). It’s conventional to square this distance to ensure that it is positive and to make our error function differentiable. 

https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png


# y = mx + b
# m is slope, b is y-intercept
def computeErrorForLineGivenPoints(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        totalError += (points[i].y - (m * points[i].x + b)) ** 2
    return totalError / float(len(points))


To run gradient descent on  error function, we first need to compute its gradient

Positive  gradient
https://www.google.co.in/search?q=gradient+meaning&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiItvjktMLWAhXDQI8KHb5dB0cQ_AUICigB&biw=1366&bih=662#imgrc=4Pk5nEtEItOx8M:


Negative  gradient
https://www.google.co.in/search?q=gradient+meaning&rlz=1C1GGRV_enIN764&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiItvjktMLWAhXDQI8KHb5dB0cQ_AUICigB&biw=1366&bih=662#imgrc=69F7rzVfxHdgVM:


To compute gradient, we will need to differentiate our error function. Since our function is defined by two parameters (m and  b), we will need to compute a partial derivative for each. These derivatives work out to be:

https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

We can initialize our search to start at any pair of m and b values (i.e., any line) and let the gradient descent algorithm march downhill on our error function towards the best line. Each iteration will update m and b to a line that yields slightly lower error than the previous iteration. The direction to move in for each iteration is calculated using the two partial derivatives

The direction to move in for each iteration is calculated using the two partial derivatives

https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

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

The learningRate variable controls how large of a step we take downhill during each iteration. If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.




Logistic Regression :produces discrete output






