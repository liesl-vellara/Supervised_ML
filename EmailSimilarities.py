from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# DATA INVESTIGATION 

emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
# checking the dataset
#print(emails)
# the datasets or emails from the library re tagged based on the content

# We are interested to find how effective our Naive Bayes classifier is at telling the difference between the emails.

# added categories to the emails which are tagged
# checking them out
#print(emails.data[5])

# chekcing hte email labels
#print(emails.target[5])

# the labels are numbers but those numbers correspond to the target
#print(emails.target_names)
# SPLITTING THE DATA
# Making the training and test sets
train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

# FORMATTING THE DATA
# Creating the CountVectorizer object
counter = CountVectorizer()

# training the object 
counter.fit(test_emails.data + train_emails.data)

# making a list of the counts of our words in our training set
train_counts = counter.transform(train_emails.data)

# making a list of the counts of our words in our test set
test_counts = counter.transform(test_emails.data)

# CREATING THE NAIVE BAYES ALGORITHM
# creating a MultinomialNB object
classifier = MultinomialNB()

# training the model using the train_counts as the parameter and the train_emails.target as the second parameter (label)
classifier.fit(train_counts, train_emails.target)

# Testing the NB classifier by printing teh .score() that take sthe test set and the test labels as the parameters
# the .score() returns the accuracy of the classifier on the test data. 
# accurary measures the percentage of classification a classifier correctly made
# the .score() will classify all the emails in the test set and compare the classifications to actual labels. It will complete the comparision and calculate the return of the accuracy.
print(classifier.score(test_counts, test_emails.target))
# the percentage is 97.23%
# for the ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], the percentage is 0.99

