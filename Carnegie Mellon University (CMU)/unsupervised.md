# Unsupervised Exercises of CMU

## The Fall 2009 10-601 Midterm

### 1. Principal Component Analysis

Plotted in the next figure are two dimensional data drawn from a multivariate Normal (Gaussian) distribution.

<img src="../resources/img/pca.png" alt="image.png" style="zoom: 50%;" />

#### 1.1 The Multivariate Gaussian

1. What is the mean of this distribution? Estimate the answer visually and round to the nearest integer.

   ​	**Answer:** 

   ​	$$E[X_1] = \mu_1 = 5 $$

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

      **Answer:** 

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

   **Answer:** *They are pushed further apart than the true means would be.*

2. Run k-means manually for the following dataset. Circles are data points and squares are the initial cluster centers. Draw cluster centers and the decision boundaries that define each cluster. Use as many pictures as you need until convergence.

   **Note:** Execute the algorithm such that if a mean has no points assigned to it, it says where it is for that iteration.

   <img src="..\resources\img\k-means.png" alt="image.png" style="zoom:75%;" />

3. Now draw (approximately) what a Gaussian mixture model of three gaussians with the same initial centers as for the k-means problem would converge to. Assume that the model puts no restrictions on the form of the covariance matrices and that EM updates both the means and covariance matrices

   <img src="..\resources\img\gmm.png" alt="image.png" style="zoom: 50%;" />

4. Is the classification given by the mixture model the same as the classification given by k-means? Why or why not?

   **Answer:** *I'd answer if I knew the start locations.*



## The 2003 final

### 1. GMM

Consider the  classification problem illustrated in the following figure. The data points in the figure are labeled, where "o"  corresponds to  class 0 and "+"  corresponds to class 1. We now estimate a GMM  consisting of 2 Gaussians, one Gaussian per  class, with the constraint that the  covariance matrices are identity matrices. The mixing proportions (class frequencies) and the means of the two Gaussians are free parameters.

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\gmm_2003.png" alt="image.png" style="zoom:60%;" />

1. Plot the maximum likelihood estimates of the means of the two Gaussians in the figure. Mark the means as points "x" and label them "0" and "1" according to class.

   **Answer:** *The means of the two Gaussians should be close to the center of mass of points*

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\gmm_2003_sol.png" alt="gmm_2003_sol" style="zoom:60%;" />

2. Based on the learned GMM, what is the probability of generating a new data point that belongs to  class 0?

   **Answer:** *0.5*

3.  How many data points are classified *incorrectly*?

   **Answer:** *3*

4. Draw the decision boundary in the same figure.

   **Answer:** *Since the two classes have the same number of points and identical covariance matrices, the decision boundary should be a straight line, which is also the orthogonal bisector of the line segment connecting the class means*

### 2. K-means Clustering

There is a set S consisting of 6 points in the plane shown as below, $a = (0, 0)$, $b = (8, 0)$, $c = (16, 0)$, $d = (0, 6)$, $e = (8, 6)$, $f = (16, 6)$. Now we run the k-means algorithm on those points with $k = 3$. The algorithm uses the Euclidean distance metric (i.e. the straight line distance between two points) to assign each point to its nearest centroid. Ties are broken in favor of the centroid to the left/down. Two definitions:

- A ***k*-starting configuration** is a subset of *k* starting points from *S* that form the initial centroids, e.g. {a, b, c}.
- A ***k*-partition** is a partition of *S* into *k* non-empty subsets, e.g. {a, b, e}, {c, d}, {f} is a 3-partition.

Clearly any *k*-partition induces a set of *k* centroids in the natural manner. A *k*-partition is called *stable* if a repetition of the *k*-means iteration with the induced centroids leaves it unchanged.

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\knn_2003.png" alt="knn_2003" style="zoom:70%;" />

1. How many 3-starting configurations are there? (Remember, a 3-starting configuration is just a subset, of size 3, of the six datapoints).

   **Answer:** *$C^3_6 = 20$*

2. Fill in the following table:

   | 3-partition            | Stable? | An example 3-starting configuration that can arrive at the 3-partition after 0 or more iterations | \# of unique 3-starting configurations that arrive at the 3-partition |
   | ---------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | {a, b, e}, {c,d}, {f}  | *N*     | *none*                                                       | *0*                                                          |
   | {a, b}, {d, e}, {c, f} | *Y*     | *{b,c,e}*                                                    | *4*                                                          |
   | {a, d}, {b, e}, {c, f} | *Y*     | *{a,b,c}*                                                    | *8*                                                          |
   | {a}, {d}, {b,c,e,f}    | *Y*     | *none*                                                       | *0*                                                          |
   | {a,b,d},{c},{e,f}      | *Y*     | *{a,c,f}*                                                    | *1*                                                          |



### The 2004 final

#### 1. Learning from labeled and unlabeled data

Consider the following figure which contains labeled (class 1 black circles class 2 hollow circles) and unlabeled (squares) data. We would like to use two methods discussed in class (re-weighting and co-training) in order to utilize the unlabeled data when training a Gaussian classifier.

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\semisupervised_2004.png" alt="semisupervised" style="zoom:67%;" />

1. How can we use co-training in this case (what are the two classifiers) ?

   **Answer:** *Co-training partitions the feature space into two separate sets and uses these sets to construct independent classifiers. Here, the most natural way is to use one classifier (a Gaussian) for the x axis and the second (another Gaussian) using the y axis.*

2. We would like to use re-weighting of unlabeled data to improve the classification performance. Reweighting will be done by placing a the dashed circle on each of the labeled data points and counting the number of unlabeled data points in that circle. Next, a Gaussian classifier is run with the new weights computed.

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\semi_2004_sol.png" alt="image.png" style="zoom:60%;" />

   1. To what class (hollow circles or full circles) would we assign the unlabeled point A is we were training a Gaussian classifier using only the labeled data points (with no re-weighting)?

      **Answer:** *Hollow class. Note that the hollow points are much more spread out and so the Gaussian learned for them will have a higher variance.*

   2. To what class (hollow circles or full circles) would we assign the unlabeled point A is we were training a classifier using the re-weighting procedure described above?

      **Answer:** *Again, the hollow class. Re-weighting will not change the result since it will be done independently for each of the two classes, and will produce very similar class centers to the ones in 1 above.*

   3. When we handle a polynomial regression problem, we would like to decide what degree of polynomial to use in order to fit a test set. The table below describes the dis-agreement between the different polynomials on unlabeled data and also the disagreement with the labeled data. Based on the method presented in class, which polynomial should we chose for this data? **Which of the two tables do you prefer?**

      | Disagreement on unlabled data |       |       |       |       |       | Disagreement on training data |
      | :---------------------------- | :---: | :---: | :---: | :---: | :---: | :---------------------------: |
      | **Degree**                    | **1** | **2** | **3** | **4** | **5** |                               |
      | 1                             |   0   |  0.3  |  0.5  |  0.6  |  0.7  |              0.4              |
      | 2                             |       |   0   |  0.2  |  0.4  |  0.5  |              0.2              |
      | 3                             |       |       |   0   |  0.2  |  0.5  |              0.1              |
      | 4                             |       |       |       |   0   |  0.3  |               0               |
      | 5                             |       |       |       |       |   0   |               0               |

      
      
      **Answer:** *The degree we would select is 3. Based on the classification accuracy, it is beneficial to use higher degree polynomials. However, as we said in class these might overfit. One way to test if they do or don’t is to check consistency on unlabeled data by requiring that the triangle inequality will hold for the selected degree. For a third degree this is indeed the case since* $u(2, 3) = 0.2 \leq l(2) + l(3) = 0.2 + 0.1$ *(where* $u(2, 3)$ *is the disagreement between the second and third degree polynomials on the unlabeled data and* $l(2)$ *is the disagreement between degree 2 in the labeled data). Similarly,* $u(1, 3) = 0.5 \leq l(1) + l(3) = 0.4 + 0.1$. In contrast, this does not hold for a fourth degree polynomial since $u(3, 4) = 0.2 > l(3) + l(4) = 0.1$.



## The 2006 Final

### 1. Dimensionality Reduction

In this problem four linear dimensionality reduction methods will be discussed. They are principal component analysis (PCA), linear discriminant analysis (LDA), canonical correlation analysis (CCA), non-negative matrix factorization (NMF).

1.  LDA reduces the dimensionality given labels by *maximizing the overall interclass variance relative to intraclass variance*. Plot the directions of the first PCA and LDA components in the following figures respectively.

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\lda_2006.png" alt="image.png" style="zoom:60%;" />

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\lda_sol_2006.png" alt="image solution" style="zoom:70%;" />

2.  In practice, each data point may have multiple vector-valued properties, e.g. a gene has its expression levels as well as the position on the genome. The goal of CCA is to reduce the dimensionality of the properties jointly. Suppose we have data points with two properties x and y, each of which is a 2-dimension vector. This 4-dimensional data is shown in the pair of figures below; different data points are shown in different gray scales. CCA finds $(u, v)$ to maximize the correlation $\hat{corr}(u^Tx)(v^Ty)$. In figure 2(b) we have given the direction of vector v, plot the vector u in figure 2(a).

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\cca_2006.png" alt="image.png" style="zoom:70%;" />

3. The goal of NMF is to reduce the dimensionality given non-negativity constraints. That is, we would like to find principle components $u_1,..., u_r$, each of which is of dimension $d > r$, such that the d-dimensional data $x \approx \sum^r_{i=1} z_iu_i$, and all entries in $x$, $z$, $u_{1:r}$ are nonnegative. NMF tends to find sparse (usually small L1 norm) basis vectors $u_i's$ . Below is an example of applying PCA and NMF on a face image. Please point out the basis vectors in the equations and give them correct labels (NMF or PCA).

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\nmf.png" style="zoom:67%;" />

### 2. Graph-Theoretic Clustering

#### Part A. Min-Cut and Normalized Cut

In this problem, we consider the 2-clustering problem, in which we have $N$ data points $x_{1:N}$ to be grouped in two clusters, denoted by $A$ and $B$. Given the $N$ by $N$ affinity matrix $W$,

- Min-Cut: minimizes $\sum_{i\in A} \sum_{j\in B} W_{ij}$;
- Normalized Cut: minimizes $\dfrac{\sum_{i\in A}\sum_{j\in B}W_{ij}}{\sum_{i\in A}\sum_{j=1}^N W_{ij}} + \dfrac{\sum_{i\in A}\sum_{j\in B}W_{ij}}{\sum_{i=1}^N\sum_{j\in B} W_{ij}}$

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\graph_clustering_2006.png" style="zoom:67%;" />

1. The data points are shown in Figure (A1) above. The grid unit is 1. Let $W_{ij} = e^{−||x_i−x_j||^2_2}$, give the clustering results of min-cut and normalized cut respectively (You may show your work in the figure directly).
2. The data points are shown in Figure (A2) above. The grid unit is 1. Let $W_{ij} = e^{\dfrac{−||x_i−x_j||^2_2}{2\sigma^2}}$, describe the clustering results of min-cut algorithm for $\sigma^2 = 50$ and $\sigma^2 = 0.5$ respectively.



## The 2008 final

### 1. Assorted Questions

1. (**True** or **False**) PCA and Spectral Clustering (such as Andrew Ng’s) perform eigen-decomposition on two different matrices. However, the size of these two matrices are the same.

   **Answer:** *False, the size of the metrices are not the same, the matrix of PCA has the size of the number of features, the matrix of Spectral Clustering has the size of the amount of records.*

### 2. Principle Component Analysis (PCA)

1. **Basic PCA**

   Given 3 data points in 2-d space, (1, 1), (2, 2) and (3, 3),

   1. What is the first principle component?

      **Answer:** *$pc = (1/\sqrt{2}, 1/\sqrt{2})' = (0.707, 0.707)'$ , (the negation is also correct)*

   2. If we want to project the original data points into 1-d space by principle component you choose, what is the variance of the projected data?

      **Answer:** *4/3 = 1.33*

   3. For the projected data in (b), now if we represent them in the original 2-d space, what is the reconstruction error?

      **Answer:** *0*

2. **PCA and SVD**

   Given 6 data points in 5-d space, (1, 1, 1, 0, 0), (−3, −3, −3, 0, 0), (2, 2, 2, 0, 0), (0, 0, 0, −1, −1), (0, 0, 0, 2, 2), (0, 0, 0, −1, −1). We can represent these data points by a $6\times 5$ matrix X, where each row corresponds to a data point:

   $X = \begin{bmatrix} 1&1&1&0&0 \\ -3&-3&-3&0&0 \\ 2&2&2&0&0 \\ 0&0&0&−1&−1 \\0&0&0&2&2 \\ 0&0&0&−1&-1 \end{bmatrix}$

1. What is the sample mean of the data set?

   **Answer:** *[0, 0, 0, 0, 0]*

2. What is SVD of the data matrix $X$ you choose?

   *hints*: The SVD for this matrix must take the following form, where $a, b, c, d, σ_1, σ_2$ are the parameters you need to decide.

   $X = \begin{bmatrix} a&0 \\ -3a&0 \\ 2a&0 \\ 0&b \\0&-2b \\ 0&b \end{bmatrix}\times \begin{bmatrix} \sigma_1&0 \\ 0&\sigma_2 \end{bmatrix}\times \begin{bmatrix} c&c&c&0&0 \\ 0&0&0&d&d \end{bmatrix}$

    **Answer:**

   - $a = \pm1/ \sqrt{14} = \pm0.267$, 
   - $b = \pm1/ \sqrt6 = \pm0.408$, 
   - $\sigma_1 = 1/(a \cdot c) = \sqrt42 = 6.48$, 
   - $\sigma_2 = 1/(b \cdot d) = \sqrt12 = 3.46$, 
   - $c = \pm1/ \sqrt3 = \pm0.577$,
   - $d = \pm1/ \sqrt2 = \pm0.707$.

3. What is first principle component for the original data points?

   **Answer:** $pc=\pm[c, c, c, 0, 0] = \pm[0.577, 0.577, 0.577, 0, 0]$ *(Intuition: First, we want to notice that the first three data points are co-linear, and so do the last three data points. And also the first three data points are orthogonal to the rest three data points. Then, we want notice that the norm of the first three are much bigger than the last three, therefor, the first pc has the same direction as the first three data points)*

4. If we want to project the original data points into 1-d space by principle component you choose, what is the variance of the projected data?

   **Answer:** *$var=\sigma^2_1/6 = 7$ (Intuition: we just the keep the first three data points, and set the rest three data points as [0, 0, 0, 0, 0] (since they are orthogonal to pc), and then compute the variance among them)*

5. For the projected data in (d), now if we represent them in the original 5-d space, what is the reconstruction error?

   **Answer:** $var=\sigma^2_2/6 = 2^1$ (*Intuition, since the first three data points are orthogonal with the rest three, here the rerr is the just the sum of the norm of the last three data points (2+8+2=12), and then divided by the total number (6) of data points, if we use average definition)*

## The 2009 Final

### 1. Short Questions

7.  Let a configuration of the k means algorithm correspond to the k way partition (on the set of instances to be clustered) generated by the clustering at the end of each iteration. Is it possible for the k-means algorithm to revisit a configuration? Justify how your answer proves that the k means algorithm converges in a finite number of steps.

   **Answer:** *Since the k means algorithm converges if the k way partition does not change in successive iterations, thus the k way partition has to change after every iteration. As the mean squared error monotonically decreases it is thus impossible to revisit a configuration. Thus eventually the k means algorithm will run out of configurations, and converge.*

8. Suppose you are given the following  pairs. You will simulate the k-means algorithm and Gaussian Mixture Models learning algorithm to identify TWO clusters in the data.

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\short_answer.png" style="zoom:60%;" />

   | Data # | x    | y    |
   | ------ | ---- | ---- |
   | 1      | 1.90 | 1.97 |
   | 2      | 1.76 | 0.84 |
   | 3      | 2.32 | 1.63 |
   | 4      | 2.31 | 2.09 |
   | 5      | 1.14 | 2.11 |
   | 6      | 5.02 | 3.02 |
   | 7      | 5.74 | 3.84 |
   | 8      | 2.25 | 3.47 |
   | 9      | 4.71 | 3.60 |
   | 10     | 3.17 | 4.96 |

   Suppose you are given initial assignment cluster center as {cluster1: #1}, {cluster2: #10} – the first data point is used as the first cluster center and the 10-th as the second cluster center. Please simulate the k-means (k=2) algorithm for ONE iteration. What are the cluster assignments after ONE interation? Assume k-means uses Euclidean distance. What are the cluster assignments until convergence? (Fill the table below)

   | Data # | Cluster Assignment after One Iteration | Cluster Assignment after convergence |
   | ------ | :------------------------------------: | :----------------------------------: |
   | 1      |                   1                    |                  1                   |
   | 2      |                   1                    |                  1                   |
   | 3      |                   1                    |                  1                   |
   | 4      |                   1                    |                  1                   |
   | 5      |                   1                    |                  1                   |
   | 6      |                   2                    |                  2                   |
   | 7      |                   2                    |                  2                   |
   | 8      |                   2                    |                  1                   |
   | 9      |                   2                    |                  2                   |
   | 10     |                   2                    |                  2                   |

9. Assume we would like to use spectral clustering to cluster n elements. We are using the k nearest neighbor method we discussed for generating the graph that would be used in the clustering procedure. Following this process:

   What is the maximum number of nodes that a single node is connected to?

   **Answer:** *n-1*

   What is the minimum number of nodes that a single node is connected to? 

   **Answer:** k

10. Can SVD and PCA produce the same projection result? Yes/No? If YES, under what condition they are the same? If NO, please explain why? (briefly) 

    **Answer**: Yes. When the data has a zero mean vector, otherwise you have to center the data first before taking SVD.

### 2. Clustering

1. Consider the dataset:

   {0, 4, 5, 20, 25, 39, 43, 44}

   Suppose we want the two top level clusters from this dataset. What will single link, complete link and average link output as the two clustering? If single link and complete link give the same 2 clusters, does it follow that average link will output the same 2 clusters? Explain.

   **Answer:**

   - Single Link: {0, 4, 5} {20, 25, 39, 43, 44}

   - Complete Link: {0, 4, 5} {20, 25, 39, 43, 44}

   - Avg Link: {0,4,5} {20,25,39,43,44} or {0,4,5,20,25}{39,43,44} – as both clustering at the final step correspond to the distance of 117/7 between clusters.

     This shows that even if single link and complete link produce the same cluster, avg link might behave differently for clustering.

2. We would like to clusters the numbers from 1 to 1024 using hierarchical clustering. We will use Euclidian distance as our distance measure. We break ties by combining the two clusters in which the lowest number resides. For example, if the distance between clusters A and B is the same as the distance between clusters C and D we would chose A and B as the next two clusters to combine if min{A,B} < min{C,D} where {A,B} are the set of numbers assigned to A and B.

   We would like to compare the results of the three linkage methods discussed in class for this dataset. For each of the three methods, specify the number of elements ( numbers) assigned to each of the two clusters defined by the root (that is, what are the sizes of the two clusters if we cut the hierarchical clustering tree at the root or in other words what are the sizes of the last two clusters that we combine).

   **Answer:**

   - Single link: 1023 + 1, clustering (((1, 2), 3), 4...)
   - Complete link: 512 + 512 (((1, 2),(3, 4), ...)
   - Average link: 512 + 512 (((1, 2),(3, 4), ...)

3. Hierarchical clustering may be bottom up or top down. Can a top down algorithm be exactly analogous to a bottom up algorithm ? Consider the following top down algorithm.

   1. Calculate the pairwise distance $d(P_i, P_j)$ between every two object $P_i$ and $P_j$ in the set of objects to be clustered and build a complete graph on the set of objects with edge weights = corresponding distances.
   2. Generate the Minimum Spanning Tree of the graph i.e. Choose the subset of edges E' with minimum sum of weights such that G' = (P,E') is a single connected tree. 
   3. Throw out the edge with the heaviest weight to generate two disconnected trees corresponding to two top level clusters.
   4. Repeat this step recursively on the lower level clusters to generate a top down clustering on the set of n objects.

   Does this top down algorithm perform analogously to any bottom up algorithm that you have encountered in class ? Why ?

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\hierarchical_clustering.png" style="zoom:70%;" />

   The clustering corresponds to single-link bottom-up clustering. The edges used to calculate the cluster distances for the single link bottom up clustering correspond to the edges of the MST (since all points must be clustered, and the cluster distance is single link and chooses the min wt edge joining together two so far unconnected clusters). Thus, the heaviest edge in the tree corresponds to the top most cluster, and so on. See example above.

### 3. Dimensionality Reduction

You have the following data: 

| data # | x     | y     |
| ------ | ----- | ----- |
| 1      | 5.51  | 5.35  |
| 2      | 20.82 | 24.03 |
| 3      | -0.77 | -0.57 |
| 4      | 19.30 | 19.38 |
| 5      | 14.24 | 12.77 |
| 6      | 9.74  | 9.68  |
| 7      | 11.59 | 12.06 |
| 8      | -6.08 | -5.22 |

You want to reduce the data into a single dimension representation. You are given the first principal component (0.694, 0.720).

1.  What is the representation (projected coordinate) for data #1 (x=5.51, y=5.35) in the first principal space?

   **Answer:** (-5.74 or -5.75)

2.  What are the x y coordinates in the original space reconstructed using this first principal representation for data #1 (x=5.51, y=5.35)?

   **Answer:** (5.31, 5.55)

3. What is the representation (projected coordinate) for data #1 (x=5.51, y=5.35) in the second principal space?

   **Answer:** 0.28 ($\pm0.28, \pm0.25$ are accepted)

4. What is the reconstruction error if you use two principal components to represent original data?

   **Answer:** 0

## The 2010 fall 601

### 1. Short Questions

5. Imagine we would like to cluster houses around Pittsburgh without using their exact addresses. For each house, we map properties of the house to a numeric value. For instance, the house’s location is mapped as Oakland = 0, Shadyside = 1, Squirrel Hill = 2, etc., the exterior material is brick = 0, aluminum = 1, wood = 2, etc., the kitchen color is white = 0, green = 1, tan = 2, etc. We have 50 such features so each house can be represented as a vector in R 50. Which of the three clustering algorithms learned in class (hierarchical clustering, k-means and Gaussian mixture models) would be most appropriate for this task? Explain briefly for each algorithm

   **Answer:** Hierarchical clustering is most appropriate. Even though we converted our categorical data to a numeric format, the mean of these vectors is meaningless so we should not use k-means. Likewise, the data does not obey a Gaussian distribution so Gaussian mixture models are a poor choice. However, hierarchical clustering with a suitable distance function can reasonably cluster this data because hierarchical clustering can handle categorical features.

### 2. Clustering

1. We would like to cluster the points in Figures 8 and 9 (which are the same) using k-means and GMM, respectively. In both cases we set k = 2. We perform several random restarts for each algorithm and chose the best one as discussed in class. For each method show the resulting cluster centers in the appropriate figure (k-means on Figure 8 and GMM on Figure 9).

   <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\knn_2010.png" style="zoom: 50%;" />

   

<img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\gmm_2010.png" style="zoom:50%;" />

 1. For the same figure (which is repeated in Figures 10 and 11) we would like to use hierarchical clustering. We will use the Euclidian distance as the distance function. In both cases we cut the tree at the second level to obtain two clusters. For two of the linkage models learned in class, single and average link, circle the resulting groups of points on each of the figures (Figure 10 - single link, Figure 11 - average link).

    <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\single_link_2010.png" style="zoom: 67%;" />

    <img src="C:\_School\Clases Uni\ML\ml-questions\resources\img\avg_link_2010.png" style="zoom:60%;" />

    