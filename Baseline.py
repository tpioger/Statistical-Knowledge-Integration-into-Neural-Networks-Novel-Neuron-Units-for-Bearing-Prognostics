import numpy as np
import pandas as pd
import model as ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tensorflow as tf

# variables
norm_value_RUL = 50000
epochs = 100
batch_size = 64
verbose = 0
slicing = 1000
bearing_dataframe = pd.read_pickle(f"dataframe_feature_bearing_pronostia_slicing_{slicing}_window_500.pkl")

runs = 1

for nb_run in range(runs):
    print(nb_run)
    true_RUL = []
    prediction_list = []
    history_list = []
    rmse_list = []
    mae_list = []
# leave one out
    for i in range(1, 7):
        print(f"LOOV {i}")
        test_selected = bearing_dataframe[bearing_dataframe["unit"] == i]
        print(np.array(test_selected).shape)
        train_selected = bearing_dataframe[bearing_dataframe["unit"] != i]

        X_train, X_val = train_test_split(train_selected, test_size=0.2, random_state=42, shuffle=True)

        train_df = X_train.copy()
        train_label = train_df['RUL'] / norm_value_RUL
        train_df = train_df.drop(['unit', 'cycle', 'RUL'], axis=1)

        val_df = X_val.copy()
        val_label = val_df['RUL'] / norm_value_RUL
        val_df = val_df.drop(['unit', 'cycle', 'RUL'], axis=1)

        test_df = test_selected.copy()
        test_label = test_df['RUL']
        true_RUL.append(test_label)
        test_df = test_df.drop(['unit', 'cycle', 'RUL'], axis=1)

        train_mean = train_df.mean(axis=0)
        train_std = train_df.std(axis=0)
        train_norm = (train_df - train_mean) / train_std
        val_norm = (val_df - train_mean) / train_std
        test_norm = (test_df - train_mean) / train_std

        model_dense_baseline = ml.dense_baseline_model()
        model_dense_baseline.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        history = model_dense_baseline.fit(train_norm, train_label,
                                       epochs=epochs,
                                       validation_data=(val_norm, val_label),
                                       batch_size=batch_size,
                                       verbose=verbose
                                       )

        history_list.append(history)
        predict_rul = model_dense_baseline.predict(test_norm)
        predict_rul *= norm_value_RUL
        prediction_list.append(predict_rul)
        rmse = np.sqrt(mean_squared_error(test_label, predict_rul))
        rmse_list.append(rmse)
        mae = mean_absolute_error(test_label, predict_rul)
        mae_list.append(mae)

        print(f"rmse : {rmse}, mae: {mae}")

    print(f"rmse:{np.mean(rmse_list)} +/- {np.std(rmse_list)}")
    print(f"Mae : {np.mean(mae_list)} +- {np.std(mae_list)}")

    save = False
    if save:
        for j in range(len(prediction_list)):
            with open(f'baseline_runs/baseline_prediction_{epochs}_epochs_slicing_{slicing}_bearing_{j + 1}_run_{nb_run + 1}_rerun_{seed}',
                      'wb') as f:
                np.save(f, prediction_list[j])

# for i in range(len(prediction_list)):
#     with open(f'true_rul_{i+1}_slicing_{slicing}.npy', 'wb') as f:
#         np.save(f, true_RUL[i])
