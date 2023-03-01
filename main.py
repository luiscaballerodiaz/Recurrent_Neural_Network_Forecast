import pandas as pd
import numpy as np
import utils
from data_visualization import DataPlot
from keras import models
from keras import callbacks
from sklearn.preprocessing import StandardScaler

# Action to perform:
#   0 --> Simulate model and calculate training and validation results
#   1 --> Load model in models_to_load and calculate test predictions
action = 1
models_to_load = ['RNN trained model 32 outputs layer, dropout=0, recurrent dropout=0.5, learning rate=0.0001 and lookback=24.h5',
                  'CNN trained model 32 outputs layer, dropout=0.5, learning rate=0.0001 and lookback=24.h5']
# General settings
lookforward = 24
batch_size = 32
train_ind = 180000
val_ind = 225000
patience_stop = 50

# Model specific settings
epochs = 80
network = 'CNN'
lookback_list = [24, 24, 24]
outputs_list = [32, 32, 32]
dropout_list = [0.5, 0.25, 0]
recurrent_dropout_list = [0.5, 0.25, 0]
learning_rate_list = [0.0001, 0.0001, 0.0001]
tag = 'definitive model sweep'

visualization = DataPlot()
pd.set_option('display.max_columns', None)
column_target = 'temp'
tag = network + ' ' + tag
sourcedf = pd.read_csv('hrly_Irish_weather.csv')
print("Sourcedata from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
timedata = sourcedf.iloc[:, 4]
sourcedf = sourcedf.iloc[:, 5:]  # remove the columns for the weather station
df = utils.replace_values(sourcedf, ['all', ' '])
visualization.target_vs_feature(dataset=df, target=column_target,
                                plot_name='Target vs feature correlation original', ncolumns=7)
df = utils.replace_values(df, ['wetb', -49], ['clht', 999])
print("Scrubbed data from CSV type: {} and shape: {}".format(type(df), df.shape))
visualization.histogram(dataset=df, plot_name='Histogram', ncolumns=7)
visualization.target_vs_feature(dataset=df, target=column_target,
                                plot_name='Target vs feature correlation', ncolumns=7)
target = np.array(df[column_target])
df_drop = df.drop(column_target, axis=1)
features = np.array(df_drop)
scaler = StandardScaler()
scaler.fit(features[:train_ind+1])  # Fit scaler ONLY in the training part
features = scaler.transform(features)  # Apply the transform to both training and testing part

if action == 0:
    description = []
    loss = []
    val_loss = []
    for i in range(len(lookback_list)):
        lookback = lookback_list[i]
        outputs = outputs_list[i]
        dropout = dropout_list[i]
        recurrent_dropout = recurrent_dropout_list[i]
        learning_rate = learning_rate_list[i]
        if network.lower() == 'cnn':
            description.append('{} outputs layer, dropout={}, learning rate={} and lookback={}'
                               .format(outputs, dropout, learning_rate, lookback))
        elif network.lower() == 'rnn':
            description.append('{} outputs layer, dropout={}, recurrent dropout={}, learning rate={} and lookback={}'
                               .format(outputs, dropout, recurrent_dropout, learning_rate, lookback))

        model = utils.create_network(network, features, outputs, recurrent_dropout, dropout, learning_rate)

        min_index_train = lookback
        max_index_train = train_ind - lookforward - 1
        train_gen = utils.generator(features, target, lookback=lookback, min_index=min_index_train,
                                    max_index=max_index_train, batch_size=batch_size, lookforward=lookforward)
        min_index_val = train_ind + lookback + 1
        max_index_val = val_ind - lookforward - 1
        val_gen = utils.generator(features, target, lookback=lookback, min_index=min_index_val,
                                  max_index=max_index_val, batch_size=batch_size, lookforward=lookforward)

        callbacks_list = callbacks.EarlyStopping(monitor='val_loss', patience=patience_stop)
        history = model.fit(train_gen, steps_per_epoch=(max_index_train - min_index_train) // batch_size, epochs=epochs,
                            callbacks=callbacks_list, validation_data=val_gen,
                            validation_steps=(max_index_val - min_index_val) // batch_size)

        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        name = network.upper() + ' trained model ' + description[i] + '.h5'
        model.save(name)

    visualization.plot_results(loss, val_loss, description, tag)

elif action == 1:
    models_list = [models.load_model(model) for model in models_to_load]
    lst = models_to_load[0].split()
    for word in lst:
        if 'lookback' in word:
            lookback = int(word[9:11])
            break
    min_index = val_ind + lookback + 1
    max_index = features.shape[0] - lookforward - 1
    preds = []
    mae_models = []
    for n, model in zip(range(len(models_list)), models_list):
        test_gen = utils.generator(features, target, lookback=lookback, min_index=min_index, max_index=max_index,
                                   batch_size=batch_size, lookforward=lookforward)
        mae_models.append(round(model.evaluate(test_gen, steps=(max_index - min_index) // batch_size), 2))
        print('MODEL {}\nTEST MAE LOSS: {}\n\n'.format(models_to_load[n], mae_models[n]))
        preds.append(model.predict(test_gen, steps=(max_index - min_index) // batch_size))
    dummy = []
    mae = 0
    for i, index in enumerate(range(min_index, max_index+1)):
        dummy.append(target[index])
        mae += abs(dummy[i] - target[index + lookforward])
    preds.append(dummy)
    mae_models.append(round(mae / (1 + max_index - min_index), 2))
    print('DUMMY MODEL\nTEST MAE LOSS: {}\n\n'.format(mae_models[-1]))
    visualization.plot_predictions(preds, mae_models, target[min_index + lookforward:max_index + lookforward],
                                   models_to_load, 5000)
    visualization.plot_predictions(preds, mae_models, target[min_index + lookforward:max_index + lookforward],
                                   models_to_load, 10000)
    visualization.plot_predictions(preds, mae_models, target[min_index + lookforward:max_index + lookforward],
                                   models_to_load, 15000)


