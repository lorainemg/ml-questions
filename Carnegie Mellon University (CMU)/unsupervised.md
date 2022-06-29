# Unsupervised Exercises of CMU

## The Fall 2009 10-601 Midterm

### 1. Principal Component Analysis

Plotted in the next figure are two dimensional data drawn from a multivariate Normal (Gaussian) distribution.

![image.png](../resources/img/pca.png)

#### 1.1 The Multivariate Gaussian

1. What is the mean of this distribution? Estimate the answer visually and round to the nearest integer.

   ​	$$ E[X_1] = \mu_1 = 5 $$

   ​	$$ E[X_2] = \mu_2 = 5 $$

2. Would the off-diagonal covariance $\sum_{1,2} = Cov (X_1, X_2)$ be:
   - [ ] negative
   - [x] positive
   - [ ] approximately zero

#### 1.2 Principal Component Analysis

Define $v_1$ and $v_2$ as the directions of the first and second principal component, with $|||v_1|| = ||v_2|| = 1$. These directions define a change of basis

$$Z_1 = (X - \mu)\cdot v_1$$

$$Z_2 = (X - \mu)\cdot v_2$$

1. Sketch and label $v_1$ and $v_2$ on the following figure (a copy of the previous figure). The arrows should originate from the mean of the distribution. You do not need to solve the SVD, instead visually estimate the directions.

   ![image.png](../resources/img/pca_answer.png)

2. The covariance $Cov(Z_1, Z_2)$, is (circle):
   - [ ] negative
   - [ ] positive
   - [x] approximately zero
3. Which point (A or B) would have the higher reconstruction error after projecting onto the first principal component direction $v_1$? Circle one:
   - [x] Point A
   - [ ] Point B