import pandas as pd
import numpy as np
from keras import layers
from keras import optimizers
from keras import models
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def replace_spaces(df):
    df.replace(to_replace=' ', value=np.nan, inplace=True)
    sourcedf_na = df.isna()
    print('Original null values: \n{}\n'.format(sourcedf_na.sum()))
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features = df.columns.values.tolist()
    preprocessor = ColumnTransformer(transformers=[('imputer', imputer, features)])
    df = preprocessor.fit_transform(df)
    df = pd.DataFrame(df, columns=features)
    sourcedf_na = df.isna()
    print('Null values after scrubbing: \n{}\n'.format(sourcedf_na.sum()))
    return df


def data_outliers_removal(df, *outliers):
    """Remove sample containing outliers"""
    for i in range(len(outliers)):
        if outliers[i][1] == '>':
            df = df.loc[df[outliers[i][0]] < outliers[i][2], :]
        elif outliers[i][1] == '>=':
            df = df.loc[df[outliers[i][0]] <= outliers[i][2], :]
        elif outliers[i][1] == '=':
            df = df.loc[df[outliers[i][0]] == outliers[i][2], :]
        elif outliers[i][1] == '<':
            df = df.loc[df[outliers[i][0]] > outliers[i][2], :]
        elif outliers[i][1] == '<=':
            df = df.loc[df[outliers[i][0]] >= outliers[i][2], :]
    return df


def generator(features_data, target_data, lookback, min_index, max_index, batch_size, lookforward):
    i = min_index
    while True:
        if (i + batch_size) > max_index:
            i = min_index + lookback
        samples = np.zeros([batch_size, lookback, features_data.shape[1]])
        target = np.zeros([batch_size])
        for j in range(i, i + batch_size):
            samples[j-i, :, :] = features_data[j-lookback:j, :]
            target[j-i] = target_data[j+lookforward]
        i += batch_size
        yield samples, target


def create_rnn(features, outputs, recurrent_dropout, dropout, learning_rate):
    model = models.Sequential()
    model.add(layers.GRU(outputs, activation='relu', input_shape=(None, features.shape[1]),
                         recurrent_dropout=recurrent_dropout))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate), loss='mae')
