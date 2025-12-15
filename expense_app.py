import pandas as pd
import joblib

model = joblib.load('expense_model.pkl')
encoder = joblib.load('expense_encoder.pkl')

salary = int(input('enter your salary'))
level = int(input('enter your level'))

sample = pd.DataFrame({
    'salary':[salary],
    'level':[level]
})

#if your datas are categorical you must convert them to small letters
#eg . sample['gender'] = sample['gender'].str.lower()

converted = encoder.transform(sample)

make_prediction  = model.predict(converted)

print('Your predicted expense is:', make_prediction)