import pandas as pd
import random
import joblib
from pandas.api.types import CategoricalDtype
from category_encoders.ordinal import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from category_encoders.m_estimate import MEstimateEncoder

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from lightautoml.ml_algo.boost_cb import BoostCB

def encoding(df):
    df_enc = df.copy()
    cat_features = df_enc.select_dtypes('object').columns.tolist()

    cities_in_train = df_enc['Город'].unique().tolist() + ['other']
    cities_dtype = CategoricalDtype(categories=cities_in_train, ordered=False)
    df_enc['Город'] = df_enc['Город'].where(df_enc['Город'].isin(cities_in_train), 'other')

    enc = OrdinalEncoder(return_df=True, handle_unknown='return_nan', handle_missing='return_nan').fit(df_enc[cat_features])

    df_enc[cat_features] = enc.transform(df_enc[cat_features])
    return df_enc

def outliers(df):
    df_outl = df.copy()
    def is_outlier(df, column):
        q25 = df[column].quantile(0.25)
        q75 = df[column].quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    outliers_columns_list = ['Площадь', 'Этаж', 'Размер_участка', 'Кво_комнат', 'Расход_тепла', 'Кво_вредных_выбросов',
                             'Кво_спален', 'Кво_ванных']

    for elem in outliers_columns_list:
        df.loc[df[elem] > is_outlier(df, elem)[1], elem] = is_outlier(df, elem)[1]

    return df_outl

def main():

    seed = random.randint(1, 1000)
    random.seed(seed)

    df = pd.read_csv('train.csv')
    df = df.drop(['id'], axis=1)

    preprocessor = Pipeline(steps=[
        ('encode', FunctionTransformer(encoding)),
        ('outliers', FunctionTransformer(outliers))
    ])

    prepr_df = preprocessor.fit_transform(df)

    TIMEOUT = 3600
    N_THREADS = 8
    CV = 4
    TARGET = ['Цена']

    roles = {'target': ['Цена'],
             'drop': ['id', 'Направление', 'Ктгр_вредных_выбросов',  'Верхний_этаж', 'Кво_фото', 'Последний_этаж',
                      'Нлч_почтового_ящика', 'Нлч_кондиционера', 'Нлч_балкона', 'Нлч_гаража']}

    TASK = Task('reg', metric='mape', greater_is_better=False)
    reader = PandasToPandasReader(TASK, cv=CV, random_state=seed)

    #pipe1
    mbie = ModelBasedImportanceEstimator()
    pipe0 = LGBSimpleFeatures()
    model0_lvl1 = BoostLGBM(default_params={'learning_rate': 0.05, 'num_leaves': 128, 'seed': seed, 'num_threads': N_THREADS})
    selector_lvl1 = ImportanceCutoffSelector(pipe0, model0_lvl1, mbie, cutoff=0)

    pipe1 = LGBSimpleFeatures()
    param_tuner1_lvl1 = OptunaTuner(n_trials=60, timeout=60)
    model1_lvl1 = BoostLGBM(default_params={'learning_rate': 0.05, 'num_leaves': 128, 'seed': seed, 'num_threads': N_THREADS})
    model2_lvl1 = BoostLGBM(default_params={'learning_rate': 0.025, 'num_leaves': 64, 'seed': seed, 'num_threads': N_THREADS})

    pipeline_lvl1 = MLPipeline([
        model0_lvl1,
        (model2_lvl1, param_tuner1_lvl1),
    ],
                              pre_selection=selector_lvl1, features_pipeline=pipe1, post_selection=None)

    #pipe2
    pipe2 = LGBSimpleFeatures()
    model0_lvl2 = BoostCB()
    param_tuner1_lvl2 = OptunaTuner(n_trials=30, timeout=60)

    pipeline_lvl2 = MLPipeline([(model0_lvl2, param_tuner1_lvl2)], pre_selection=None, features_pipeline=pipe2, post_selection=None)

    #pipe3
    pipe3 = LGBSimpleFeatures()
    model0_lvl3 = BoostLGBM(
        default_params={'learning_rate': 0.05, 'num_leaves': 128, 'seed': seed, 'num_threads': N_THREADS})

    pipeline_lvl3 = MLPipeline([model0_lvl3],
                               pre_selection=None, features_pipeline=pipe3, post_selection=None)

    automl = AutoML(reader, [[pipeline_lvl1], [pipeline_lvl2], [pipeline_lvl3]],
                    skip_conn=False)

    pipe_pred = automl.fit_predict(prepr_df, roles=roles, verbose=3)
    print(f'train score: {mape(prepr_df[TARGET].values, pipe_pred.data[:, 0])}')

    joblib.dump(automl, 'model.pkl')

if __name__ == '__main__':
    main()

