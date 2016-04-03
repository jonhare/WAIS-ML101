*WAIS AWAY DAY 2016*

# Machine Learning 101 - Tutorial

## Introduction and Acknowledgements
This tutorial is largely based on the official ["Working With Text Data" scikit-learn tutorial](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). A number of changes have been made to ensure that it fits the format required for the away-day session, and we've added additional bits that demonstrate how to cluster data with k-means. Through this tutorial you'll learn how to:

* Load the a dataset of newsgroup posts
* Extract feature vectors suitable for machine learning
* Use K-Means and hierarchical agglomerative clustering to automatically group data
* Train a classifier to that predicts what newsgroup a post belongs to
* Use a grid search strategy to find a good configuration of both the feature extraction components and the classifier

## Prerequisites
To use this tutorial you'll use the Python language with the `scikit-learn` package to perform feature extraction, clustering and classification tasks. You'll need access to a computer with the following installed:

- `Python` (> 2.6)
- `NumPy` (>= 1.6.1)
- `SciPy` (>= 0.9)
- `scikit-learn` (>= 0.17.0)

The easiest way to install all of these together is with [Anaconda](https://www.continuum.io/downloads) on Windows or Linux. On a Mac, open a terminal and run `pip install -U numpy scipy scikit-learn` (note: don't do this on Linux as there is not an official binary release & this will end up building a non-optimised version from source). You'll also need a programmer's text editor (e.g. [Sublime](http://www.sublimetext.com)).

Finally, you'll need the data set we'll be using and skeleton solutions to the exercises. If you've borrowed a laptop from us, then you can find these on the desktop in the `WAIS-ML101/sklearn-tutorial` folder. If you're using your own laptop, then you can copy this off the memory sticks we've provided, or you can clone it from GitHub using:  ...

## A data set for experimentation

For the purposes of this tutorial we're going to play with a dataset of internet newsgroup posts. The dataset is called "Twenty Newsgroups". Here is the official description, quoted from the [website](http://people.csail.mit.edu/jrennie/20Newsgroups/):

> The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, it was originally collected by Ken Lang, probably for his paper "Newsweeder: Learning to filter netnews," though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.

Start by navigating to the `WAIS-ML101/sklearn-tutorial/data/twenty_newsgroups` folder and look at how the data set is structured. There are two folders - one with data for training models, and one for testing how well a model works. Within each of the training and testing folders are 20 folders representing the 20 different newsgroups. Within these folders are the actual messages posted on the newsgroups, with one file per message. Spend some time to open a few of the files in a text editor to see their contents.

Now open a python interpreter (either `python` or `ipython`) to get started learning how to use `scikit-learn`.

We're going to start by loading the dataset into memory. `scikit-learn` contains a number of tools that can help us do this. In order to get faster execution times for the initial parts of this tutorial we will work on a partial dataset with only 4 categories out of the 20 available in the dataset:

```python
>>>
>>> categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 
... 'sci.med']
```

We can now load the list of files matching those categories using the [`sklearn.datasets.load_files`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files) function as follows (change the path to match where your copy of the data is stored):

```python
>>> from sklearn.datasets import load_files
>>> twenty_train = load_files('/path/to/data/twenty_newsgroups/train', 
... categories=categories, shuffle=True, random_state=42, encoding='latin1')
```

The returned dataset is a `scikit-learn` "bunch": a simple holder object with fields that can be both accessed as python `dict` keys or `object` attributes for convenience, for instance the `target_names` holds the list of the requested category names:

```python
>>>
>>> twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
```

The files themselves are loaded in memory in the `data` attribute. For reference the filenames are also available:
```python
>>>
>>> len(twenty_train.data)
2257
>>> len(twenty_train.filenames)
2257
```

Let’s print the first lines of the first loaded file:
```python
>>>
>>> print("\n".join(twenty_train.data[0].split("\n")[:3]))
From: sd345@city.ac.uk (Michael Collier)
Subject: Converting images to HP LaserJet III?
Nntp-Posting-Host: hampton

>>> print(twenty_train.target_names[twenty_train.target[0]])
comp.graphics
```

Supervised learning algorithms will require a category label for each document in the training set. In this case the category is the name of the newsgroup which also happens to be the name of the folder holding the individual documents.

For speed and space efficiency reasons `scikit-learn` loads the target attribute as an array of integers that corresponds to the index of the category name in the `target_names` list. The category integer id of each sample is stored in the `target` attribute:

```python
>>>
>>> twenty_train.target[:10]
array([1, 1, 3, 3, 3, 3, 3, 2, 2, 2])
```

It is possible to get back the category names as follows:

```python
>>>
>>> for t in twenty_train.target[:10]:
...     print(twenty_train.target_names[t])
...
comp.graphics
comp.graphics
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
soc.religion.christian
sci.med
sci.med
sci.med
```

You can notice that the samples have been shuffled randomly (with a fixed random number generator seed); this is useful if you select only the first samples to quickly train a model and get a first idea of the results before re-training on the complete dataset later.

## Building a Basic "Bag of Words" Feature Extractor

In order to perform machine learning on text documents, we first need to turn the text content into numerical featurevectors.

### Bags of words

The most intuitive way to do so is the "bags of words" representation:

1. Assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
2. For each document `#i`, count the number of occurrences of each word `w` and store it in `X[i, j]` as the value of feature `#j` where `j` is the index of word `w` in the dictionary

The bags of words representation implies that `n_features` is the number of distinct words in the corpus: this number is typically larger that 100,000.

If `n_samples == 10000`, storing `X` as a numpy array of type float32 would require 10000 x 100000 x 4 bytes = **4GB in RAM** which is barely manageable on today's computers.

Fortunately, **most values in X will be zeros** since for a given document less than a couple thousands of distinct words will be used. For this reason we say that bags of words are typically **high-dimensional sparse datasets**. We can save a lot of memory by only storing the non-zero parts of the feature vectors in memory.

`scipy.sparse` matrices are data structures that do exactly this, and `scikit-learn` has built-in support for these structures.

### Tokenizing text with `scikit-learn`

Text preprocessing, tokenizing and filtering of stop-words are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors:

```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> count_vect = CountVectorizer()
>>> X_train_counts = count_vect.fit_transform(twenty_train.data)
>>> X_train_counts.shape
(2257, 35788)
```

CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices:

```python
>>> count_vect.vocabulary_.get(u'algorithm')
4690
```

The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

### From occurrences to frequencies

Occurrence count is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics. To avoid these potential discrepancies, it suffices to divide the number of occurrences of each word in a document by the total number of words in the document. The number of times a term occurs in a document, divided by the number of terms in a document is called the **term frequency** (*tf*).

Another refinement on top of *tf* is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus. In order to achieve this we can weight terms on the basis of the **inverse document frequency** (*idf*). The _document frequency_ is the number of documents a given word occurs in; the inverse document frequency is often defined as the number of documents divided by the *df*.

Combining *tf* and *idf* results in a *family* of weightings (*tf* is usually multiplied by *idf*, but there a few different variations of how *idf* is computed) known as "Term Frequency - Inverse Document Frequency" ([tf–idf](https://en.wikipedia.org/wiki/Tf–idf)).

Both **tf** and **tf–idf** can be computed using `scikit-learn` as follows:

```python
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
>>> X_train_tf = tf_transformer.transform(X_train_counts)
>>> X_train_tf.shape
(2257, 35788)
```

In the above code, we firstly use the `fit(..)` method to fit our estimator to the data and secondly the `transform(..)` method to transform our count-matrix to a tf-idf representation. These two steps can be combined to achieve the same end result faster by skipping redundant processing. This is done through using the `fit_transform(..)` method as shown below, and as mentioned in the note in the previous section:

```python
>>> tfidf_transformer = TfidfTransformer()
>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
>>> X_train_tfidf.shape
(2257, 35788)
```

Rather than transforming the raw counts with the `TfidfTransformer`, it is alternatively possible to use the `TfidfVectorizer` to directly parse the dataset. The advantage of doing this is that it can automatically filter out less informative words on the basis of stop-words, document frequency, etc.:

```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> tfidf_vect = TfidfVectorizer(stop_words='english',max_df=0.5,min_df=2)
>>> X_train_tfidf = tfidf_vect.fit_transform(twenty_train.data)
>>> X_train_tfidf.shape
(2257, 18189)
```

As you can see from the output, the number of features was reduced from 35788 to 18189 using this approach.

## Exploring clustering using scikit-learn

Now we've extracted features from our training documents, we're in a position to experiment with clustering.

### Getting started with K-Means

```python
>>> from sklearn.cluster import KMeans
>>> km = KMeans(4)
>>> km.fit(X_train_tfidf)
```

```python
>>> order_centroids = km.cluster_centers_.argsort()[:, ::-1]
>>> terms = tfidf_vect.get_feature_names()
>>> for i in range(4):
...     print "Cluster %d:" % i,
...     for ind in order_centroids[i, :10]:
...         print ' %s' % terms[ind],
...     print
...
Cluster 0:  com  graphics  university  posting  host  nntp  msg  article  thanks  know
Cluster 1:  keith  caltech  livesey  sgi  wpd  solntze  schneider  jon  cco  morality
Cluster 2:  god  jesus  people  bible  believe  christian  christians  think  say  don
Cluster 3:  pitt  geb  banks  gordon  cs  cadre  dsl  shameful  n3jxp  surrender
```

```python
from sklearn import metrics
print "Homogeneity: %0.3f" % metrics.homogeneity_score(twenty_train.target, km.labels_)
```

---------------------------------------

> **Exercise:** Can you print out which cluster each document belongs to? Hint: use `km.predict(X_train_tfidf)` to get the cluster assignment of each document index, and `twenty_train.filenames` to get the filenames of the corresponding documents.

---------------------------------------


## Building a predictive model

## K-Nearest neighbour classification

## Extension exercises

### Better features

### Other types of clustering

### More advanced classifiers


