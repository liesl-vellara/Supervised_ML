import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Each row in these DataFrames corresponds to a single pitch that the batter saw in the 2017 season. To begin, let’s take a look at all of the features of a pitch.
#print(aaron_judge.columns)
#print(aaron_judge.description.unique())
#print(aaron_judge.type.unique())
# 'S' for strike
# 'B; for ball
# 'X; for neither (can be a hit or an out)

# changing 'S' to 1
# changine 'B' to 0
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})
# investigating the updated dataframe
print(aaron_judge.type.unique())
# there is NaN present in place of 'X'

# we want to know if a pitch is a ball or a strike on its location over the plate
# columns plate_x and plate_z handles that
#print(aaron_judge['plate_x'])
# plate_x measures how far left or right the pitch is from the center of the home plate
# if plate_x = 0, that means teh pitch was directly in the middle of the home plate.

#plate_z measures how high off the ground the pitch was
# if plate_z = 0, that means the pitch was at ground level when it got to the home plate

# Removing all the NaN in every row of plate_x, plate_z and type
aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
#print(aaron_judge.type.unique())

# using .scatter from plt with five parameters

fig, ax = plt.subplots()
plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap = plt.cm.coolwarm, alpha = 0.25)
#ax.set_ylim(-2, 2)
# creating the Support Vector Machine model
# we will split the data into train and test data
training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

# creating the Support Vector Machine model
classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)
classifier.fit(training_set[['plate_x', 'plate_z']],training_set['type'])
draw_boundary(ax, classifier)
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

plt.show()
plt.close()

# CREATING A FUNCTION FOR DIFFERENT DATA FRAME

def support_vector_machine(df):
  df['type'] = df['type'].map({'S': 1, 'B': 0})
  df = df.dropna(subset = ['type', 'plate_x', 'plate_z'])
  fig, ax = plt.subplots()
  plt.scatter(df.plate_x, df.plate_z, c = df.type, cmap = plt.cm.coolwarm, alpha = 0.25)
  training_set, validation_set = train_test_split(df,random_state = 1)
  model = SVC(kernel = 'rbf', gamma = 3, C = 1)
  model.fit(training_set[['plate_x', 'plate_z']],training_set['type'])
  draw_boundary(ax, model)
  plt.show()
  plt.close()
  return "This is the score of the dataframe " + str(model.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


print(support_vector_machine(jose_altuve))
print(support_vector_machine(david_ortiz))

