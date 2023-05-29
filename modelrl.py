  
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

 
    # Read the wine-quality csv file from the URL
   
data = pd.read_csv("datarl", sep=';')

    
 
    # Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)
 
    # The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]


alpha = 0.67#à changer
l1_ratio = 0.3#à changer


lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)
 
predicted_qualities = lr.predict(test_x)
 
rmse = mean_absolute_error(test_y, predicted_qualities,squared=False)
mae = mean_squared_error(test_y, predicted_qualities)
r2 = r2_score(test_y, predicted_qualities)
 
print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))

print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)