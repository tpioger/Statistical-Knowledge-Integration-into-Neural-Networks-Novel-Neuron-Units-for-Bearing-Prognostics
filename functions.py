import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def load_dataset_and_rul():
    """
    Load the dataset and the rul and return them as a numpy array format
    :return:
    list_bearing: numpy array
    list_bearing_rul: numpy array
    """
    path = os.getcwd()
    path = path + "/" + 'data/' + 'bearing_and_rul/'
    bearing = np.load(path + 'list_bearing.npy', allow_pickle=True)
    bearing_rul = np.load(path + 'list_bearing_rul.npy', allow_pickle=True)
    return bearing, bearing_rul


def leave_one_bearing_out(bearing, bearing_rul, index, slicing_bool=True, slicing_value=100):
    """
    Leave one bearing out and his corresponding RUL
    :param bearing: Nested array representing all 6 Bearings of PRONOSTIA
    :param bearing_rul: Nested array representing the Remaining Useful Life of the 6 bearings of PRONOSTIA
    :param index: Index used to split leave one bearing as the test set
    :param slicing_bool: True or False. If set to True the bearing will be sliced by the slicing value. By Default True
    :param slicing_value: int. By default 100. Represent the value by which the bearing value are going to be sliced
    :return:
    """
    bearing_train = np.array([])
    bearing_rul_train = np.array([])
    for i in range(len(bearing)):
        if i != index:
            if slicing_bool:
                bearing_train = np.append(bearing_train, bearing[i][::slicing_value])
                bearing_rul_train = np.append(bearing_rul_train, bearing_rul[i][::slicing_value])
            else:
                bearing_train = np.append(bearing_train, bearing[i])
                bearing_rul_train = np.append(bearing_rul_train, bearing_rul[i])

    if slicing_bool:
        bearing_test = bearing[index][::slicing_value]
        bearing_rul_test = bearing_rul[index][::slicing_value]
    else:
        bearing_test = bearing[index]
        bearing_rul_test = bearing_rul[index]
    return bearing_train, bearing_rul_train, bearing_test, bearing_rul_test


def normalisation(bearing_train, bearing_rul_train, bearing_test, bearing_rul_test):
    """
    :param bearing_train: numpy array. Train set Bearing
    :param bearing_rul_train: numpy array. Train set RUl
    :param bearing_test: numpy array. Test set Bearing
    :param bearing_rul_test: numpy array. Test set RUL
    :return:
     Numpy array. Normalized value for ML
    """
    mean = bearing_train.mean()
    std = bearing_train.std()

    bearing_train_norm = (bearing_train - mean) / std
    bearing_test_norm = (bearing_test - mean) / std

    bearing_rul_train_norm = bearing_rul_train / 500
    bearing_rul_test_norm = bearing_rul_test / 500

    return bearing_train_norm, bearing_rul_train_norm, bearing_test_norm, bearing_rul_test_norm


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window_rul(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.min(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides), axis=1)


def split_sequence(sequence, rul, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], np.min(rul[i:end_ix])
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_data(data, rul, interval=500):
    newData, newRUL = list(), list()
    split, splitRUL = split_sequence(data, rul, interval)
    newData.extend(split)
    newRUL.extend(splitRUL)
    return np.array(newData, dtype=np.float32), np.array(newRUL, dtype=np.float32)


def feature_extraction_all(bearing_window):
    # Kamal features: 5
    feature_peak_value, feature_impulse_factor, feature_crest_factor, feature_shape_factor, feature_clearance_factor = np.array(
        []), np.array(
        []), np.array([]), np.array([]), np.array([])
    # Statistical features: 9
    feature_max, feature_min, feature_max_min, feature_mean, feature_var, feature_std, feature_rms, feature_skewness, feature_kurtosis = np.array(
        []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array(
        [])

    # Frequency features: 5
    feature_pmm, feature_frequencies_max, feature_frequencies_mean, feature_frequencies_var, feature_frequencies_sum = np.array(
        []), np.array([]), np.array([]), np.array([]), np.array([])

    # feature_frequencies_kurtosis = []
    # feature_frequencies_skewness = []

    # Total features : 19
    for i in range(len(bearing_window)):
        # time series
        # kamal
        peak_to_peak = np.abs(np.max(bearing_window[i]) - np.min(bearing_window[i]))
        feature_peak_value = np.append(feature_peak_value, peak_to_peak)
        impulse_factor = np.max(bearing_window[i]) / np.mean(np.abs(bearing_window[i]))
        feature_impulse_factor = np.append(feature_impulse_factor, impulse_factor)
        crest_factor = np.max(bearing_window[i]) / np.sqrt(np.mean(np.square(bearing_window[i])))
        feature_crest_factor = np.append(feature_crest_factor, crest_factor)
        shape_factor = np.sqrt(np.mean(np.square(bearing_window[i]))) / np.mean(np.abs(bearing_window[i]))
        feature_shape_factor = np.append(feature_shape_factor, shape_factor)
        clearance_factor = np.max(bearing_window[i]) / np.mean(np.sqrt(np.abs(bearing_window[i])))
        feature_clearance_factor = np.append(feature_clearance_factor, clearance_factor)
        # statistical

        maximum = np.max(bearing_window[i])
        feature_max = np.append(feature_max, maximum)
        minimum = np.min(bearing_window[i])
        feature_min = np.append(feature_min, minimum)
        mean = np.mean(bearing_window[i])
        feature_mean = np.append(feature_mean, mean)
        variance = np.var(bearing_window[i])
        feature_var = np.append(feature_var, variance)
        # max_min = maximum - minimum
        # feature_max_min = np.append(feature_max_min, max_min)
        std = np.std(bearing_window[i])
        feature_std = np.append(feature_std, std)
        root_mean_square = np.sqrt(np.mean(np.square(bearing_window[i])))
        feature_rms = np.append(feature_rms, root_mean_square)
        skewness = stats.skew(bearing_window[i])
        feature_skewness = np.append(feature_skewness, skewness)
        kurtosis = stats.kurtosis(bearing_window[i])
        feature_kurtosis = np.append(feature_kurtosis, kurtosis)

        # frequency features

        fft = np.fft.fft(bearing_window[i])
        magnitude_spectrum = np.abs(fft)
        pmm = np.max(magnitude_spectrum) / np.mean(magnitude_spectrum)
        feature_pmm = np.append(feature_pmm, pmm)
        power_spectrum = np.abs(fft) ** 2 / len(fft)
        maximum_ps = np.max(power_spectrum)
        feature_frequencies_max = np.append(feature_frequencies_max, maximum_ps)
        feature_frequencies_mean = np.append(feature_frequencies_mean, np.mean(power_spectrum))
        feature_frequencies_var = np.append(feature_frequencies_var, np.var(power_spectrum))
        feature_frequencies_sum = np.append(feature_frequencies_sum, np.sum(power_spectrum))

    # reshape the features
    feature_max = feature_max.reshape((feature_max.shape[0], 1))
    feature_max = feature_max.astype('float32')
    feature_min = feature_min.reshape((feature_min.shape[0], 1))
    feature_min = feature_min.astype('float32')
    feature_mean = feature_mean.reshape((feature_mean.shape[0], 1))
    feature_mean = feature_mean.astype('float32')
    feature_var = feature_var.reshape((feature_var.shape[0], 1))
    feature_var = feature_var.astype('float32')
    feature_std = feature_std.reshape((feature_std.shape[0], 1))
    feature_std = feature_std.astype('float32')
    feature_rms = feature_rms.reshape((feature_rms.shape[0], 1))
    feature_rms = feature_rms.astype('float32')
    feature_skewness = feature_skewness.reshape((feature_skewness.shape[0], 1))
    feature_skewness = feature_skewness.astype('float32')
    feature_kurtosis = feature_kurtosis.reshape((feature_kurtosis.shape[0], 1))
    feature_kurtosis = feature_kurtosis.astype('float32')

    feature_peak_value = feature_peak_value.reshape((feature_peak_value.shape[0], 1))
    feature_peak_value = feature_peak_value.astype('float32')

    feature_impulse_factor = feature_impulse_factor.reshape((feature_impulse_factor.shape[0], 1))
    feature_impulse_factor = feature_impulse_factor.astype('float32')

    feature_crest_factor = feature_crest_factor.reshape((feature_crest_factor.shape[0], 1))
    feature_crest_factor = feature_crest_factor.astype('float32')

    feature_shape_factor = feature_shape_factor.reshape((feature_shape_factor.shape[0], 1))
    feature_shape_factor = feature_shape_factor.astype('float32')

    feature_clearance_factor = feature_clearance_factor.reshape((feature_clearance_factor.shape[0], 1))
    feature_clearance_factor = feature_clearance_factor.astype('float32')
    # Frequency features

    feature_frequencies_max = feature_frequencies_max.reshape((feature_frequencies_max.shape[0], 1))
    feature_frequencies_max = feature_frequencies_max.astype('float32')

    feature_frequencies_mean = feature_frequencies_mean.reshape((feature_frequencies_mean.shape[0], 1))
    feature_frequencies_mean = feature_frequencies_mean.astype('float32')

    feature_frequencies_var = feature_frequencies_var.reshape((feature_frequencies_var.shape[0], 1))
    feature_frequencies_var = feature_frequencies_var.astype('float32')

    feature_frequencies_sum = feature_frequencies_sum.reshape((feature_frequencies_sum.shape[0], 1))
    feature_frequencies_sum = feature_frequencies_sum.astype('float32')

    feature_pmm = feature_pmm.reshape((feature_pmm.shape[0], 1))
    feature_pmm = feature_pmm.astype('float32')

    # concatenate all feature
    feature_train = np.concatenate((feature_max,
                                    feature_min,
                                    feature_mean,
                                    feature_var,
                                    feature_std,
                                    feature_rms,
                                    feature_skewness,
                                    feature_kurtosis,
                                    feature_peak_value,
                                    feature_impulse_factor,
                                    feature_crest_factor,
                                    feature_shape_factor,
                                    feature_clearance_factor,
                                    feature_frequencies_max,
                                    feature_frequencies_mean,
                                    feature_frequencies_var,
                                    feature_frequencies_sum,
                                    feature_pmm), axis=1)

    return feature_train


def save_alpha_lambda_single(rul_true, rul_predict, correct, l, title, save_file_name):
    """

    :param rul_true:
    :param rul_predict:
    :param correct:
    :param l:
    :param title:
    :param save_file_name:
    :return:
    """
    rul_true = np.array(rul_true)
    rul_true = np.reshape(rul_true, (rul_true.shape[0],))
    rul_true_high_interval_list, rul_true_low_interval_list = list(), list()
    for k in range(len(rul_true)):
        high_interval = rul_true[k] * (1 + l)
        low_interval = rul_true[k] * (1 - l)
        rul_true_high_interval_list.append(high_interval)
        rul_true_low_interval_list.append(low_interval)

    fig, ax = plt.subplots()

    interval_1 = rul_true[0] * 0.1
    interval_2 = rul_true[0] * 0.2
    interval_3 = rul_true[0] * 0.3
    interval_4 = rul_true[0] * 0.4
    interval_5 = rul_true[0] * 0.5
    interval_6 = rul_true[0] * 0.6
    interval_7 = rul_true[0] * 0.7
    interval_8 = rul_true[0] * 0.8
    interval_9 = rul_true[0] * 0.9
    interval_10 = rul_true[0]

    list_interval = [interval_10, interval_9, interval_8, interval_7, interval_6, interval_5, interval_4, interval_3,
                     interval_2, interval_1, 0]

    y = np.max(rul_true_high_interval_list)
    if y < np.max(rul_predict):
        y = np.max(rul_predict)

    y += 200

    for iteration in range(10):
        ax.annotate(correct[iteration], xy=(list_interval[iteration], y), color='black')

    ax.axvline(0, color='lightgray', linestyle='--')
    ax.axvline(interval_1, color='lightgray', linestyle='--')
    ax.axvline(interval_2, color='lightgray', linestyle='--')
    ax.axvline(interval_3, color='lightgray', linestyle='--')
    ax.axvline(interval_4, color='lightgray', linestyle='--')
    ax.axvline(interval_5, color='lightgray', linestyle='--')
    ax.axvline(interval_6, color='lightgray', linestyle='--')
    ax.axvline(interval_7, color='lightgray', linestyle='--')
    ax.axvline(interval_8, color='lightgray', linestyle='--')
    ax.axvline(interval_9, color='lightgray', linestyle='--')
    ax.axvline(interval_10, color='lightgray', linestyle='--')

    ax.set_title("{0}".format(title))
    ax.plot(rul_true, rul_true_high_interval_list, color='black', linestyle='--')
    ax.plot(rul_true, rul_true_low_interval_list, color='black', linestyle='--')
    ax.fill_between(rul_true, rul_true_high_interval_list, rul_true_low_interval_list, alpha=0.2)
    ax.plot(rul_true, rul_true, color='black', linestyle='-')

    ax.plot(rul_true, rul_predict, c='black', label='Feature inside of the model')

    ax.invert_xaxis()
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    plt.savefig(f'figures/{save_file_name}')

    return None


def correct_alpha_lamba(prediction_array, true_array, l):
    # Split array in 10 sub array
    prediction_array_split = np.array_split(prediction_array, 10)
    true_array_split = np.array_split(true_array, 10)
    correct_list = []
    for i in range(len(prediction_array_split)):
        correct = 0
        for j in range(len(prediction_array_split[i])):
            value_predicted = prediction_array_split[i][j]
            true_value = true_array_split[i][j]
            interval_high = true_value * (1 + l)
            interval_low = true_value * (1 - l)

            if interval_high >= value_predicted >= interval_low:
                correct += 1
        correct_percentage = (correct / len(prediction_array_split[i])) * 100
        correct_percentage = np.round(correct_percentage, 2)
        correct_list.append(correct_percentage)

    return correct_list


def moving_average(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array t o
    # consider every window of size 3
    while i < len(arr) - window_size + 1:
        # Calculate the average of current window
        window_average = round(np.sum(arr[
                                      i:i + window_size]) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages

def rmse_mae_calculation(true_rul, prediction):
    rmse_list = []
    mae_list = []
    for i in range(len(true_rul)):
        rmse = np.sqrt(mean_squared_error(true_rul[i], prediction[i]))
        mae = mean_absolute_error(true_rul[i], prediction[i])
        rmse_list.append(rmse)
        mae_list.append(mae)
    return rmse_list, mae_list


def create_pandas_df(rmse, mae):
    columns = []
    for i in range(len(rmse)):
        name = f"bearings_{i + 1}"
        columns.append(name)
    df = pd.DataFrame((np.round(rmse), np.round(mae)), columns=columns)
    df['mean'] = df.mean(axis=1).round(0)
    df['std'] = df.std(axis=1).round(0)
    return df
