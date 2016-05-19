Regression
---

### Lecture Overview

| Week | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Week 1](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-1-simple-linear-regression) | Simple Linear Regression |
| [Week 2](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-2-multiple-regression) | Multiple Regression |
| [Week 3](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-3-accessing-performance) | Accessing Performance |
| [Week 4](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-4-ridge-regression) | Ridge Regression |
| [Week 5](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-5-feature-selection--lasso) | Feature Selection & Lasso |
| [Week 6](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression#week-6-nearest-neighbors--kernel-regression) | Nearest Neighbors & Kernel Regression |


## Week 1: Simple Linear Regression

**- Notes: Since the minimum of a function is derivative = 0**

**+ Approach 1: closed form solution (normal equation): set the gradient (vector of derivative) = 0 and solve the equation, immediately converge at the local minimum.**

**+ Approach 2: gradient descent: while not converged (derivative &lt; epsilon), update the coefficient w to go descent and converge at the local minimum**

Aside: The python notation x.xxe+yy means x.xx * 10^(yy). e.g 100 = 10^2 = 1*10^2 = 1e2

**- Lecture**

- [Lecture 1](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/lecture/week1)
- [PhillyCrime.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week1/PhillyCrime.ipynb)
- [Linear Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week1/Linear%20Regression.ipynb)
- [quiz - Simple Linear Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week1/quiz%20-%20Simple%20Linear%20Regression.ipynb)

**- Assignment**

- [Assignment 1](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week1)
- [week-1-simple-regression-assignment-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week1/week-1-simple-regression-assignment-exercise.ipynb)
- [quiz - assignment1.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week1/quiz%20-%20assignment1.ipynb)


## Week 2: Multiple Regression

**- Lecture**

- [Lecture 2](https://github.com/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week2/week2_multipleregression-annotated.pdf)
- [Multiple Regression.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week2/Multiple%20Regression.ipynb)
- [quiz - Multiple Regression.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week2/quiz%20-%20Multiple%20Regression.ipynb)

**- Numpy Tutorial**

- [numpy-tutorial.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week2/numpy-tutorial.ipynb)

**- Assignment**

- [Assignment 2](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week2)
- [week-2-multiple-regression-assignment-1-exercise.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week2/week-2-multiple-regression-assignment-1-exercise.ipynb)
- [quiz - week2-assignment1.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week2/quiz%20-%20week2-assignment1.ipynb)
- [week-2-multiple-regression-assignment-2-exercise.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week2/week-2-multiple-regression-assignment-2-exercise.ipynb)
- [quiz - week2-assignment2.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week2/quiz%20-%20week2-assignment2.ipynb)


## Week 3: Accessing Performance

**- Lecture**

- [Lecture 3](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/lecture/week3)
- [Assessing Performance.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week3/Assessing%20Performance.ipynb)
- [quiz - Assessing Performance.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week3/quiz%20-%20Assessing%20Performance.ipynb)

**- Assignment**

- [Assignment 3](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week3)
- [week-3-polynomial-regression-assignment-exercise.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week3/week-3-polynomial-regression-assignment-exercise.ipynb)
- [quiz - week3-assignment.ipynb](http://nbviewer.ipython.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week3/quiz%20-%20week3-assignment.ipynb)


## Week 4: Ridge Regression

**- Lecture**

- [Lecture 4](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/lecture/week4)
- [Ridge Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week4/Ridge%20Regression.ipynb)
- [quiz - Ridge Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week4/quiz%20-%20Ridge%20Regression.ipynb)

**- Assignment**

- [Assignment 4](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week4)
- [week-4-ridge-regression-assignment-1-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week4/week-4-ridge-regression-assignment-1-exercise.ipynb)
- [quiz - week4-assignment1.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week4/quiz%20-%20week4-assignment1.ipynb)
- [week-4-ridge-regression-assignment-2-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week4/week-4-ridge-regression-assignment-2-exercise.ipynb)
- [quiz - week4-assignment2.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week4/quiz%20-%20week4-assignment2.ipynb)


## Week 5: Feature Selection & Lasso

**- Lecture**

- [Lecture 5](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/lecture/week5)
- [Feature Selection & Lasso.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week5/Feature%20Selection%20%26%20Lasso.ipynb)
- [Overfitting_Demo_Ridge_Lasso.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week5/Overfitting_Demo_Ridge_Lasso.ipynb)
- [quiz - Feature Selection and Lasso.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week5/quiz%20-%20Feature%20Selection%20and%20Lasso.ipynb)

**- Assignment**

- [Assignment 5](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week5)
- [week-5-lasso-assignment-1-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week5/week-5-lasso-assignment-1-exercise.ipynb)
- [quiz - week5-assignment1.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week5/quiz%20-%20week5-assignment1.ipynb)
- [week-5-lasso-assignment-2-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week5/week-5-lasso-assignment-2-exercise.ipynb)
- [quiz - week5-assignment2.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week5/quiz%20-%20week5-assignment2.ipynb)


## Week 6: Nearest Neighbors & Kernel Regression

**- Lecture**

- [Lecture 6](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/lecture/week6)
- [Nearest Neighbors & Kernel Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week6/Nearest%20Neighbors%20%26%20Kernel%20Regression.ipynb)
- [quiz - Nearest Neighbors & Kernel Regression.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/lecture/week6/quiz%20-%20Nearest%20Neighbors%20%26%20Kernel%20Regression.ipynb)

**- Assignment**

- [Assignment 6](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/2_regression/assignment/week6)
- [week-6-local-regression-assignment-exercise.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week6/week-6-local-regression-assignment-exercise.ipynb)
- [quiz-week6-assignment.ipynb](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/2_regression/assignment/week6/quiz-week6-assignment.ipynb)
