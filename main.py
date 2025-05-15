import pandas as pd
import joblib
import numpy as np

model = joblib.load('predictor_model.pkl')
le_result = joblib.load('label_encoder_result.pkl')
le_l = joblib.load('label_encoder_l.pkl')
le_n = joblib.load('label_encoder_n.pkl')

daily_df = pd.read_excel('Daily__02152025.xlsx', header=None)

daily_df = daily_df[[0, 7, 11, 3, 13]].dropna()
daily_df.columns = ['Player', 'Result', 'PlanetsCombo', 'Intensity', 'PlanetsIntensity']

daily_df['PlanetsCombo'] = le_l.transform(daily_df['PlanetsCombo'].astype(str).apply(lambda x: x if x in le_l.classes_ else le_l.classes_[0]))
daily_df['PlanetsIntensity'] = le_n.transform(daily_df['PlanetsIntensity'].astype(str).apply(lambda x: x if x in le_n.classes_ else le_n.classes_[0]))
daily_df['Intensity'] = pd.to_numeric(daily_df['Intensity'], errors='coerce').fillna(0)

X_new = daily_df[['PlanetsCombo', 'Intensity', 'PlanetsIntensity']]
predictions = model.predict(X_new)
daily_df['Predicted_Result'] = le_result.inverse_transform(predictions)

daily_df[['Player', 'Predicted_Result']].to_excel('Predicted_Output.xlsx', index=False)
