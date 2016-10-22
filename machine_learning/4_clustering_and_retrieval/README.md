Clustering & Retrieval
---

## Lecture Overview

| Week | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Week 1](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-1-welcome) | Welcome |
| [Week 2](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-2-nearest-neighbor-search) | Nearest Neighbor Search |
| [Week 3](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-3-clustering-with-k-means) | Clustering with k-means |
| [Week 4](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-4-mixture-models) | Mixture Models |
| [Week 5](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-5-mixed-membership-modeling-via-latent-dirichlet-allocation) | Mixed Membership Modeling via Latent Dirichlet Allocation |
| [Week 6](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval#week-6-hierarchical-clustering--closing-remarks) | Hierarchical Clustering & Closing Remarks |


## Week 1: Welcome

**[Lecture](./lecture/week1)**
- [intro](./lecture/week1/01_slides-presented-in-this-module_intro.pdf)
- [Software tools](./lecture/week1/Software%20tools.pdf)

## Week 2: Nearest Neighbor Search

- [Lecture](./lecture/week2)
- [Assignment](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval/assigment/week2)

#### Introduction to nearest neighbor search and algorithms

- [retrieval-intro-annotated](./lecture/week2/01_slides-presented-in-this-module_retrieval-intro-annotated.pdf)

#### The importance of data representations and distance metrics

- __Lecture__
	- quiz-Representations and metrics.ipynb [[ipynb](./lecture/week2/quiz-Representations%20and%20metrics.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week2/quiz-Representations%20and%20metrics.ipynb)]

- __Assignment__
	- 0_nearest-neighbors-features-and-metrics_graphlab.ipynb [[ipynb](./assignment/week2/0_nearest-neighbors-features-and-metrics_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week2/0_nearest-neighbors-features-and-metrics_graphlab.ipynb)]
	- quiz-week2-assignment1.ipynb [[ipynb](./assignment/week2/quiz-week2-assignment1.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week2/quiz-week2-assignment1.ipynb)]

#### Scaling up k-NN search using KD-trees

- __Lecture__
	- quiz-KD-trees.ipynb [[ipynb](./lecture/week2/quiz-KD-trees.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week2/quiz-KD-trees.ipynb)]

#### Locality sensitive hashing for approximate NN search

- __Lecture__
	- quiz-Locality Sensitive Hashing.ipynb [[ipynb](./lecture/week2/quiz-Locality%20Sensitive%20Hashing.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week2/quiz-Locality%20Sensitive%20Hashing.ipynb)]

- __Assignment__
	- 1_nearest-neighbors-lsh-implementation_graphlab.ipynb [[ipynb](./assignment/week2/1_nearest-neighbors-lsh-implementation_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week2/1_nearest-neighbors-lsh-implementation_graphlab.ipynb)]
	- quiz-week2-assignment2.ipynb [[ipynb](./assignment/week2/quiz-week2-assignment2.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week2/quiz-week2-assignment2.ipynb)]


## Week 3: Clustering with k-means

- [Lecture](./lecture/week3)
- [Assignment](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval/assigment/week3)

#### Introduction to clustering

- [kmeans-annotated](./lecture/week3/01_slides-presented-in-this-module_kmeans-annotated.pdf)

#### Clustering via k-means

- __Lecture__
	- quiz-K-means.ipynb [[ipynb](./lecture/week3/quiz-k-means.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week3/quiz-k-means.ipynb)]

- __Assignment__
	- 2_kmeans-with-text-data_graphlab.ipynb [[ipynb](./assigment/week3/2_kmeans-with-text-data_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week3/2_kmeans-with-text-data_graphlab.ipynb)]
	- quiz-week3-assignment1.ipynb [[ipynb](./assigment/week3/quiz-week3-assignment1.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week3/quiz-week3-assignment1.ipynb)]

#### MapReduce for scaling k-means

- __Lecture__
	- quiz-MapReduce for k-means.ipynb [[ipynb](./lecture/week3/quiz-MapReduce%20for%20k-means.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week3/quiz-MapReduce%20for%20k-means.ipynb)]

## Week 4: Mixture Models

- [Lecture](./lecture/week4)
- [Assignment](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval/assigment/week4)

#### EM for Gaussian mixtures

- [EM-annotated](./lecture/week4/01_slides-presented-in-this-module_mixmodel-EM-annotated.pdf)
- quiz-EM for Gaussian mixtures.ipynb [[ipynb](./lecture/week4/quiz-EM%20for%20Gaussian%20mixtures.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week4/quiz-EM%20for%20Gaussian%20mixtures.ipynb)]

#### Implementing EM for Gaussian mixtures

- __Assignment__
	- 3_em-for-gmm_graphlab.ipynb [[ipynb](./assigment/week4/3_em-for-gmm_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week4/3_em-for-gmm_graphlab.ipynb)]
	- quiz-week4-assignment1.ipynb [[ipynb](./assigment/week4/quiz-week4-assignment1.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week4/quiz-week4-assignment1.ipynb)]

#### Clustering text data with Gaussian mixtures

- __Assignment__
	- 4_em-with-text-data_graphlab.ipynb [[ipynb](./assigment/week4/4_em-with-text-data_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week4/4_em-with-text-data_graphlab.ipynb)]
	- quiz-week4-assignment2.ipynb [[ipynb](./assigment/week4/quiz-week4-assignment2.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week4/quiz-week4-assignment2.ipynb)]

## Week 5: Mixed Membership Modeling via Latent Dirichlet Allocation

- [Lecture](./lecture/week5)
- [Assignment](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval/assigment/week5)

#### Introduction to latent Dirichlet allocation

- __Lecture__
	- [LDA-annotated](./lecture/week5/01_slides-presented-in-this-module_LDA-annotated.pdf)
	- quiz-Latent Dirichlet Allocation.ipynb [[ipynb](./lecture/week5/quiz-Latent%20Dirichlet%20Allocation.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week5/quiz-Latent%20Dirichlet%20Allocation.ipynb)]

#### Bayesian inference via Gibbs sampling

#### Collapsed Gibbs sampling for LDA

#### Summarizing latent Dirichlet allocation

- __Lecture__
	- quiz-Learning LDA model via Gibbs sampling.ipynb [[ipynb](./lecture/week5/quiz-Learning%20LDA%20model%20via%20Gibbs%20sampling.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/lecture/week5/quiz-Learning%20LDA%20model%20via%20Gibbs%20sampling.ipynb)]

- __Assignment__
	- 5_lda_graphlab.ipynb [[ipynb](./assigment/week5/5_lda_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week5/5_lda_graphlab.ipynb)]
	- quiz-week5-assignment.ipynb [[ipynb](./assigment/week5/quiz-week5-assignment.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week5/quiz-week5-assignment.ipynb)]

## Week 6: Hierarchical Clustering & Closing Remarks

- [Lecture](./lecture/week6)
- [Assignment](https://github.com/tuanavu/coursera-university-of-washington/tree/master/machine_learning/4_clustering_and_retrieval/assigment/week6)

#### Hierarchical clustering and clustering for time series segmentation

- __Lecture__
	- [closing-annotated.pdf](./lecture/week6/closing-annotated.pdf)	

- __Assignment__
	- 6_hierarchical_clustering_graphlab.ipynb [[ipynb](./assigment/week6/6_hierarchical_clustering_graphlab.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week6/6_hierarchical_clustering_graphlab.ipynb)]
	- quiz-week6-assignment.ipynb [[ipynb](./assigment/week6/quiz-week6-assignment.ipynb)] [[nbviewer](http://nbviewer.jupyter.org/github/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week6/quiz-week6-assignment.ipynb)]

