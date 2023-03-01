import pandas as pd
import numpy as np
from keras import layers
from keras import optimizers
from keras import models
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def replace_values(df, *changes):
    """Replace the values in the corresponding column for the mean value of the column"""
    for i in range(len(changes)):
        if 'all' in changes[i][0]:
            df.replace(to_replace=changes[i][1], value=np.nan, inplace=True)
        else:
            df.replace(to_replace={changes[i][0]: changes[i][1]}, value=np.nan, inplace=True)
    print('Original null values: \n{}\n'.format(df.isna().sum()))
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    features = df.columns.values.tolist()
    preprocessor = ColumnTransformer(transformers=[('imputer', imputer, features)])
    df = preprocessor.fit_transform(df)
    df = pd.DataFrame(df, columns=features)
    print('Null values after scrubbing: \n{}\n'.format(df.isna().sum()))
    return df


def generator(features_data, target_data, lookback, min_index, max_index, batch_size, lookforward):
    i = min_index
    while True:
        if (i + batch_size) > max_index:
            i = min_index
        samples = np.zeros([batch_size, lookback, features_data.shape[1]])
        target = np.zeros([batch_size])
        for j in range(i, i + batch_size):
            samples[j-i, :, :] = features_data[j-lookback:j, :]
            target[j-i] = target_data[j+lookforward-1]
        i += batch_size
        yield samples, target


def create_network(nn_type, features, outputs, recurrent_dropout, dropout, learning_rate):
    model = models.Sequential()
    if nn_type.lower() == 'rnn':
        model.add(layers.GRU(outputs, activation='relu', input_shape=(None, features.shape[1]),
                             recurrent_dropout=recurrent_dropout))
    elif nn_type.lower() == 'cnn':
        model.add(layers.Conv1D(outputs, 5, activation='relu', input_shape=(None, features.shape[1])))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(outputs, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(learning_rate=learning_rate), loss='mae')
    return model
