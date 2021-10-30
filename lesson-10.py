import pandas as pd
import numpy as np

from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.tasks.common_metric import mean_quantile_error, mean_absolute_percentage_error



def transform_str_fields(data, fields):
    str_to_numbers = {'A': 0, 'B': 1}
    for field in fields:
        data[field] = data[field].replace(str_to_numbers)

    return data


def transform_rooms(data):
    data.loc[train_df['Rooms'] == 0, 'Rooms'] = 1
    data.loc[train_df['Rooms'] > 6, 'Rooms'] = 6

    return data


def transfrom_square(data):
    quartilies_square = np.percentile(train_df['Square'], [25, 50, 75])
    mu = quartilies_square[1]
    sig = 0.74 * (quartilies_square[2] - quartilies_square[0])
    data.loc[train_df['Square'] > mu + 3 * sig, 'Square'] = mu + 3 * sig
    data.loc[train_df['Square'] == 0, 'Square'] = data['Square'].mean()

    return data


def transfrom_kitchen_square(data):
    quartilies_kitchensquare = np.percentile(train_df['KitchenSquare'], [25, 50, 75])
    mu = quartilies_kitchensquare[1]
    sig = 0.74 * (quartilies_kitchensquare[2] - quartilies_kitchensquare[0])
    data.loc[train_df['KitchenSquare'] > mu + 3 * sig, 'KitchenSquare'] = mu + 3 * sig
    data.loc[train_df['KitchenSquare'] < 3, 'KitchenSquare'] = data['KitchenSquare'].mean()

    return data


def transform_life_square(data):
    data.loc[data['LifeSquare'] > data['Square'], 'LifeSquare'] = data['Square'] - data['KitchenSquare']

    return data


def transform_year(data):
    data.loc[data['HouseYear'] > 2021, 'HouseYear'] = 2021

    return data


def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
    print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
    print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.fillna(0, inplace=True)
train_df = transform_rooms(train_df)
train_df = transform_str_fields(train_df, ['Ecology_2', 'Ecology_3', 'Shops_2'])
train_df = transfrom_square(train_df)
train_df = transfrom_kitchen_square(train_df)
train_df = transform_life_square(train_df)
train_df = transform_year(train_df)

test_df.fillna(0, inplace=True)
test_df = transform_rooms(test_df)
test_df = transform_str_fields(test_df, ['Ecology_2', 'Ecology_3', 'Shops_2'])
test_df = transfrom_square(test_df)
test_df = transfrom_kitchen_square(test_df)
test_df = transform_life_square(test_df)
test_df = transform_year(test_df)

TASK = Task('reg', loss='mse', metric=mean_absolute_percentage_error, greater_is_better=False)
TIMEOUT = 300000
N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TARGET_NAME = 'Price'
TEST_SIZE = 0.2

roles = {'target': TARGET_NAME, 'drop': ['Id']}
automl_model = TabularAutoML(task=TASK,
                             timeout=TIMEOUT,
                             cpu_limit=N_THREADS,
                             gpu_ids='all',
                             reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
                             general_params={'use_algos': [['lgb_tuned', 'cb_tuned', 'cb'], ['lgb_tuned', 'cb']]},
                             tuning_params={'max_tuning_iter': 10},
                             )
oof_pred = automl_model.fit_predict(train_df, roles = roles)
test_df['Price'] = automl_model.predict(test_df).data
test_df.to_csv('result.csv', columns=['Id', 'Price'], index=False)

X = train_df
y = train_df[TARGET_NAME]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=21)
oof_pred = automl_model.fit_predict(X_train, roles = roles)
y_valid_preds = automl_model.predict(X_valid).data
print(f'R2 = {r2(y_valid.values, y_valid_preds)}')
