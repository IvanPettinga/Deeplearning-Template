import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1.0 Preparing the Data
# 1.1 Reading the data
dataset = pd.read_csv("life_expectancy.csv")

print(dataset.head(5))
print(dataset.describe())
for col in dataset.columns:
  print(col)
# 1.2 prediction and feature columns
dataset = dataset.drop(columns =["Country"])
labels = dataset.iloc[:,-1] 
features = dataset.iloc[:,1:20] 
print(labels)
for col in features.columns:
  print(col)

# 1.3 Create dummies for the features
features = pd.get_dummies(features)

# 1.4 Train-test-split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=23)

# 1.5 select the numerical features
numerical_features = features.select_dtypes(include =
['float64', 'int64']) #make function for selecting numericals
numerical_columns = numerical_features.columns #Apply function to the columns of the features dataset to get the column names with numerical features

# 1.6 make a function called ct where these column names are applied to scale a dataset
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder = 'passthrough')

# 1.7 Apply the function to the features dataset
features_train = ct.fit_transform(features_train) #This function is used to determine the scaling
features_test = ct.transform(features_test) #avoid info leakage apply train function

# 2.0 Building the model

# 2.1 Create Sequential model in Keras
my_model = Sequential()
# 2.2 Add layers
my_model.add(InputLayer(input_shape = (features_train.shape[1],))) #input layer
my_model.add(Dense(64,activation="relu")) #magic layer
my_model.add(Dense(1)) #output layer

# 2.3 summary of model design
print(Sequential.summary(my_model))

#3.0 initilization and compilation choose a optimizer 
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = "mse", metrics = ["mae"], optimizer = opt)

# 4.0 fit the model
my_model.fit(features_train, labels_train, epochs = 240, batch_size = 1, verbose = 1)

# 5.0 Evaluate the model
mse,mae = my_model.evaluate(test_train,test_labels, verbose = 0)
print(mse)
print(mae)

predicted_values = my_model.predict(features_test) 
print(r2_score(labels_test, predicted_values)) 





























