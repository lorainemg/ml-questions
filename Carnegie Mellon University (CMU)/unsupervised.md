# Unsupervised Exercises of CMU

## The Fall 2009 10-601 Midterm

### 1. Principal Component Analysis

Plotted in the next figure are two dimensional data drawn from a multivariate Normal (Gaussian) distribution.

<img src="../resources/img/pca.png" alt="image.png" style="zoom: 50%;" />

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

   <img src="../resources/img/pca_answer.png" alt="image.png" style="zoom: 67%;" />

2. The covariance $Cov(Z_1, Z_2)$, is (circle):
   - [ ] negative
   - [ ] positive
   - [x] approximately zero
3. Which point (A or B) would have the higher reconstruction error after projecting onto the first principal component direction $v_1$? Circle one:
   - [x] Point A
   - [ ] Point B



## The 2004 10-701 midterm

### 1. K-Means and Hierarchical Clustering

1. Perform K-means on the dataset given below. Circles are data points and there are two initial cluster centers, at data points 5 and 7. Draw the cluster centers (as squares) and the decision boundaries that define each cluster. If no points belong to a particular cluster, assume its center does not change. Use as many of the pictures as you need for convergence.

   ![image.png](..\resources\img\kmeans.png)

   2. Give one advantage of hierarchical clustering over K-means clustering, and one advantage of K-means over hierarchical clustering.

      *Some advantages of hierarchical clustering:*

      - *Don't need to know hoy many clusters you're after*
      - *Can cut hierarchy at any level to get any number of clusters*
      - *Easy to interpret hierarchy for particular applications*
      - *Can deal with long stringy data*

      *Some advantages of K-means clustering:*

      - *Can be much faster than hierarchical clustering, depending on data*
      - *Nice theoretical framework*
      - *Can incorporate new data and reform clusters easily*

## The 2001 final

### 1. Clustering

In the left of the following pictures I show a dataset. In the right figure I sketch the globally maximally likely mixture of three Gaussians for the given data.

- Assume we have a protective code in place that prevents any degenerate solutions in which some Gaussian grows infinitesimally small.
- Assume a GMM model in which all parameters (class probabilities, class centroids and class covariances) can be varied.

<img src="..\resources\img\gaussians.png" alt="image.png" style="zoom:80%;" />1. Using the same notation and the same assumptions, sketch the globally maximally likely mixture of **two** Gaussians.

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\2_gaussians.png" alt="image.png" style="zoom:67%;" />



2. Using the same notation and the same assumptions, sketch a mixture of three distinct Gaussians that is stuck in a suboptimal configuration (i.e. in which infinitely many more interations of the EM algorithm would remain in essentially the same suboptimal configuration). (*You must not give an answer in which two or more Gaussians all have the same mean vectors --we are looking for an answer in which all Gaussians have distinct mean vectors*)

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\2_gaussians.png" alt="image.png" style="zoom:67%;" />

3. Using the same notation and the same assumptions, sketch the globally maximally likely mixture of two Gaussians in the following new, dataset.

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\other_gaussian.png" alt="image.png" style="zoom:75%;" />

4. Now, suppose we ran k-means with $k=2$ on this dataset. Show the rough locations of the centers of the two clusters in the configuration with globally minimal distortion

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\other_gaussian.png" alt="image.png" style="zoom:75%;" />

## The 2002 final

### 1. K-means and Gaussian Mixture Models

1. What is the effect on the means found by k-means (as opposed to the true means) of overlapping clusters?

   *They are pushed further apart than the true means would be.*

2. Run k-means manually for the following dataset. Circles are data points and squares are the initial cluster centers. Draw cluster centers and the decision boundaries that define each cluster. Use as many pictures as you need until convergence.

   **Note:** Execute the algorithm such that if a mean has no points assigned to it, it says where it is for that iteration.

   <img src="..\resources\img\k-means.png" alt="image.png" style="zoom:75%;" />

3. Now draw (approximately) what a Gaussian mixture model of three gaussians with the same initial centers as for the k-means problem would converge to. Assume that the model puts no restrictions on the form of the covariance matrices and that EM updates both the means and covariance matrices

   <img src="..\resources\img\gmm.png" alt="image.png" style="zoom: 50%;" />

   4. Is the classification given by the mixture model the same as the classification given by k-means? Why or why not?

      *I'd answer if I knew the start locations.*



## The 2003 final