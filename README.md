# AML
Applied Machine Learning Projects at Cornell with Professor Serge Bolongie

Assignment 1:

1.Implemented K-neartest-neighbor algorithm from scratch
2.Trained a kNN model to recognize sketches of digits and return identity of new digits in the testing data set
3.Performed cross-validation using Naive Bayes Classifier & Logistic Regression Model. Used data from training set (demographics, type of cabin, etc.), to train the algorithm and then evaluated whether an unseen list of passengers survived the Titanic disaster or not.

Assignment 2:

*Using the Yale Eigenfaces database, we calculated an “average” face, performed a Singular Value Decomposition, found the low-rank approximation, plotted the rank-r approximation error, represented the 2500-dimensional face image using an r-dimensional feature vector and used logistic regression to classify features from the top r vectors.
*Using Kaggle’s “What’s Cooking” recipe database, we represented each dish by a binary ingredient feature vector, used Naive Bayes Classifier to predict cuisine using recipe ingredients, calculated the Naive Bayes’ accuracy assuming Gaussian and Bernoulli priors, used Logistic Regression to predict cuisine using recipe ingredients, and finally used our Bernoulli-prior Naive Bayes to compete in the Kaggle contest.

Assignment 3:

*This assignment performs sentiment analysis of online reviews on Amazon, Yelp and IMDB. We preprocessed the written data (punctuation stripping, lemmatization, etc), represented each of the 3 review collections in a Bag of Words model, normalized the Bag of Words, implemented K-means to divide the training set into a “positive” and a “negative” cluster, and then used a logistic regression model to predict the review’s sentiment. We then ran the same process using a 2-gram n-model instead of Bag of Words; we used a logistic regression model. Finally, we implemented PCA to reduce the dimensions of features and then implemented Bag of Words on the reduced model.
*We used the Old Faithful Geyser dataset, which contains 272 observation of the geyser’s eruption time and waiting time. We implemented a Gaussian Mixture Model(GMM) and calculated how many iterations were necessary for a convergence of datapoint around different covariance means. Finally, we compared a K-means algorithm to our EMM and found that it required less iterations than the GMM to cluster the data.

Assignment 4:

*We used a random forest algorithm in order to process an image of the Mona Lisa. We then iterated the number of decision trees and size of depth in order to understand how information is represented in random forests
