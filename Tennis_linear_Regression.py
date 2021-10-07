import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
#(df.head())

# perform exploratory analysis here:
plt.scatter(df[['BreakPointsOpportunities']], df[['Winnings']], alpha = 0.4)
plt.title("Winnings in USD VS BreakPointOpportunities")
plt.xlabel("Break Point Opportunies")
plt.ylabel("Winnings in USD")
plt.show()
plt.close()
# There is a strong positive correlation between BreakPointOppurtunies and Winnings

# Wins and BreakPointsOpportunities
# we will also check with the wins and breakpoint opp
plt.scatter(df[['BreakPointsOpportunities']], df[['Wins']], alpha = 0.4)
plt.title("Wins VS BreakPointOpportunities")
plt.xlabel("Break Point Opportunies")
plt.ylabel("Wins")
plt.show()
plt.close()
# there is also another strong positive correlation between wins and breakpoint oppurtunies

# Losses and BreakPointsOpportunities
plt.scatter(df[['BreakPointsOpportunities']], df[['Losses']], alpha = 0.4)
plt.title("Losses VS BreakPointOpportunities")
plt.xlabel("Break Point Opportunies")
plt.ylabel("Losses")
plt.show()
plt.close()
# There is a strong positive correlation between losses and Break point opportunity

# perform single feature linear regressions here:
feature = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

# splitting the data to train and test using train_test_split module
feature_train, feature_test, outcome_train, outcome_test = train_test_split(feature, outcome, train_size = 0.8)

# creating a model with the LinearRegression
model = LinearRegression()

# Fitting the data to the model
model.fit(feature_train, outcome_train)

#Score the model we used (finding the mean squared method of the training set)
model.score(feature_test, outcome_test)

# predict the outcome of our model 
outcome_predict = model.predict(feature_test)

# plotting the predicted outcome through a scatter plot
plt.scatter(df[['FirstServeReturnPointsWon']], df[['Winnings']],alpha = 0.4, marker = 'o')
plt.scatter(feature_test, outcome_predict, alpha = 0.4, marker = 'x')
plt.xlabel("First Serve Return Points Won")
plt.ylabel("Winnings in USD")
plt.show()
plt.close()

# breakpointopportunies and Winnings
x = df[['BreakPointsOpportunities']]
y = df[['Winnings']]
#print(len(x))
#print(len(y))
# splitting the data to train and test using train_test_split module
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# creating a model with the LinearRegression
mlr = LinearRegression()

# Fitting the data to the model
mlr.fit(x_train, y_train)

#Score the model we used (finding the mean squared method of the training set)
mlr.score(x_test, y_test)

# predict the outcome of our model 
y_predict = mlr.predict(x_test)

# plotting the predicted outcome through a scatter plot
plt.scatter(df[['BreakPointsOpportunities']], df[['Winnings']],alpha = 0.4, marker = 'o')
plt.scatter(x_test, y_predict, alpha = 0.4, marker = 'x')
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings in USD")
plt.show()
plt.close()


# perform two feature linear regressions here:
# wins and losses in x
# BreakPointOpportunities in y

x_1 = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
y_1 = df[['Winnings']]

# spltting the data to train and test
x1_train, x1_test, y1_train, y1_test = train_test_split(x_1, y_1, train_size = 0.8)

# initialising the LinearRegression model
reg = LinearRegression()

# training the data to the model
reg.fit(x1_train, y1_train)

# getting the mean squares of the data
reg.score(x1_test, y1_test)

# predicting the outcome
y1_predicted = reg.predict(x1_test)

# the data has too many variables, so the outcome predicted has to be plotted against the outcome test data
plt.scatter(y1_test, y1_predicted, alpha = 0.4, marker = 'o')
plt.xlabel('Winnings Test')
plt.ylabel('Winnings Predicted')
plt.title('Plotting the Test data against the Predicted based on two critera')
plt.show()
plt.close()


# perform multiple feature linear regressions here:
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

# Training and testing the data
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

# fitting the training data to the model
model.fit(features_train, outcome_train)

# getting the squared mean method of the trained data and test it against the test data
model.score(features_test, outcome_test)

# predicting the outcome from the test data
outcome_predict = model.predict(features_test)


# the data has too many variables, so the outcome predicted has to be plotted against the outcome test data
plt.scatter(outcome_test, outcome_predict)
plt.xlabel('Winnings Test')
plt.ylabel('Winnings Predicted')
plt.title('Plotting the Test data against the Predicted based on 18 criteras')
plt.show()
plt.close()
