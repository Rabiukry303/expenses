import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

data = pd.read_csv('expense.csv')

#FEATURE AND TARGET
X = data[['salary','level']]
y = data['expense']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

#scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#TRAIN
#model = LinearRegression()
model = DecisionTreeRegressor()
model.fit(X_train_scaled, y_train)

print('Model successfully trained')

#EVALUATION
print('R square Score:', r2_score(y_test, model.predict(X_test_scaled)))
print('MAE           :', mean_absolute_error(y_test, model.predict(X_test_scaled)))

#sAVE MODEL
joblib.dump(model, 'expense_model.pkl')

#save encoder
joblib.dump(scaler, 'expense_encoder.pkl')

print('model and encoder saved')

salary = int(input('enter your salary'))
level = int(input('enter your level'))

sample = pd.DataFrame({
    'salary':[salary],
    'level':[level]
})

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print('predicted expense:', prediction[0])