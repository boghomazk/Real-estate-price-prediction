import pandas as pd
import random
import joblib
from train import encoding, outliers
from sklearn.pipeline import Pipeline, FunctionTransformer


seed = random.randint(1, 1000)
random.seed(seed)

test_df = pd.read_csv('public_test.csv')
test_id = test_df['id']

preprocessor = Pipeline(steps=[
        ('encode', FunctionTransformer(encoding)),
        ('outliers', FunctionTransformer(outliers))
    ])

prep_df = preprocessor.transform(test_df)
with open('model.pkl', 'rb') as file:
    model = joblib.load(file)
test_pred = model.predict(prep_df)
pd.DataFrame({'id': test_id, 'Цена': test_pred.data[:, 0]}).set_index('id').to_csv('test_pred.csv')
