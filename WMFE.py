from functions import load_dataset_and_rul, rolling_window, rolling_window_rul
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import model as ml
import tensorflow as tf

# variables
params = dict(
    norm_value_RUL=50000,
    epochs=100,
    batch_size=64,
    verbose=0,
    slicing=1000,
    window=500,
)
bearing, bearing_rul = load_dataset_and_rul()

true_RUL = []
rmse_list = []
mae_list = []
opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001)

prediction_list = []

for i in range(len(bearing)):
    print(f"LOOV: {i + 1}")
    # rolling window on RUL
    test = rolling_window(bearing[i][::params["slicing"]], params["window"])
    test_rul = rolling_window_rul(bearing_rul[i][::params["slicing"]], params["window"])
    test_rul = test_rul.reshape(test_rul.shape[0], 1)
    true_RUL.append(test_rul)
    train = []
    train_rul = []
    for k in range(6):
        if k != i:
            train.extend(bearing[k][::params["slicing"]])
            train_rul.extend(bearing_rul[k][::params["slicing"]])
    train = np.array(train).astype('float32')
    train_rul = np.array(train_rul).astype('float32')
    train = rolling_window(train, params["window"])
    train_rul = rolling_window_rul(train_rul, params["window"])

    train_mean = np.mean(train)
    train_std = np.std(train)

    train_norm = (train - train_mean) / train_std
    train_rul_norm = train_rul / params["norm_value_RUL"]
    test_norm = (test - train_mean) / train_std
    test_rul_norm = test_rul / params["norm_value_RUL"]
    # shuffle the lists with same order
    train_norm, train_rul_norm = shuffle(train_norm, train_rul_norm, random_state=0)
    split_index = int(0.8 * len(train_norm))
    train_norm, train_rul_norm, val_norm, val_rul_norm = (train_norm[:split_index], train_rul_norm[:split_index],
                                                          train_norm[split_index:], train_rul_norm[split_index:])
    model = ml.model_custom_block_stat(input_shape=params["window"], use_bias=True)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    history_custom = model.fit(train_norm, train_rul_norm,
                               batch_size=params["batch_size"],
                               epochs=params["epochs"],
                               validation_data=(val_norm, val_rul_norm),
                               verbose=params["verbose"])

    rul_predicted = model.predict(test)

    rul_predicted *= params["norm_value_RUL"]
    prediction_list.append(rul_predicted)

    rmse = np.sqrt(mean_squared_error(test_rul, rul_predicted))
    rmse_list.append(rmse)
    mae = mean_absolute_error(test_rul, rul_predicted)
    mae_list.append(mae)

    print(f"rmse : {rmse}, mae: {mae}")

print(f"rmse:{np.mean(rmse_list)} +/- {np.std(rmse_list)}")
print(f"Mae : {np.mean(mae_list)} +- {np.std(mae_list)}")

save = True
if save:
    for i in range(len(prediction_list)):
        with open(f'predictions/WMFE/prediction_multiple_features_bias_{params["epochs"]}_epochs_slicing_{params["slicing"]}_bearing_{i+1}.npy', 'wb') as f:
            np.save(f, prediction_list[i])
