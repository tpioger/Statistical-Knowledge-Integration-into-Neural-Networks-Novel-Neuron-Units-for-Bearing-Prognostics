from functions import load_dataset_and_rul, rolling_window, rolling_window_rul
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import model as ml
import tensorflow as tf
from sklearn.model_selection import train_test_split

# variables
value_rul_norm = 50000
slicing = 1000
window = 500
epochs = 100
batch_size = 64
verbose = 0
runs = 1
bearing, bearing_rul = load_dataset_and_rul()


opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001)

for nb_run in range(runs):
    print(nb_run)

    true_RUL = []
    rmse_list = []
    prediction_list = []
    mae_list = []
    history_list = []
    for i in range(len(bearing)):
        print(f"LOOV: {i + 1}")
        # rolling window sur RUL
        test = rolling_window(bearing[i][::slicing], window)
        print(test.shape)
        test_rul = rolling_window_rul(bearing_rul[i][::slicing], window)
        print(test_rul.shape)
        test_rul = test_rul.reshape(test_rul.shape[0], 1)
        true_RUL.append(test_rul)
        train = []
        train_rul = []
        for k in range(6):
            if k != i:
                train.extend(bearing[k][::slicing])
                train_rul.extend(bearing_rul[k][::slicing])
        train = np.array(train).astype('float32')
        train_rul = np.array(train_rul).astype('float32')
        train = rolling_window(train, window)
        train_rul = rolling_window_rul(train_rul, window)

        train_mean = np.mean(train)
        train_std = np.std(train)

        train_norm = (train - train_mean) / train_std
        train_rul_norm = train_rul / value_rul_norm
        test_norm = (test - train_mean) / train_std
        test_rul_norm = test_rul / value_rul_norm

        train_norm, val_norm, train_rul_norm, val_rul_norm = train_test_split(train_norm, train_rul_norm,
                                                                              test_size=0.2, random_state=42,
                                                                              shuffle=True)
        model = ml.model_custom_block_stat(input_shape=window,use_bias=False)
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        history_custom = model.fit(train_norm, train_rul_norm,
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(val_norm, val_rul_norm),
                               verbose=verbose)

        history_list.append(history_custom)
        print(test.shape)
        rul_predicted = model.predict(test)
        print(rul_predicted.shape)

        rul_predicted *= value_rul_norm
        prediction_list.append(rul_predicted)
        print(test_rul.shape)
        rmse = np.sqrt(mean_squared_error(test_rul, rul_predicted))
        rmse_list.append(rmse)
        mae = mean_absolute_error(test_rul, rul_predicted)
        mae_list.append(mae)

        print(f"rmse : {rmse}, mae: {mae}")

    print(f"rmse:{np.mean(rmse_list)} +/- {np.std(rmse_list)}")
    print(f"Mae : {np.mean(mae_list)} +- {np.std(mae_list)}")
    save = False
    if save:
        for j in range(len(prediction_list)):
            with open(f'MFE_runs/MFE_prediction_{epochs}_epochs_slicing_{slicing}_bearing_{j+1}_run_{nb_run+1}', 'wb') as f:
                np.save(f, prediction_list[j])
