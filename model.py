from tensorflow import keras
import tensorflow as tf
import layers

def dense_baseline_model():
    """
    Baseline Model Dense for RUL prediction

    Returns
    -------
    model: tensorflow model
        Tenssorflow model.

    """
    inputs = keras.Input(shape=(18,), name="inputs")
    x = keras.layers.Dense(64, activation="relu", name="first_hidden_layer")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="second_hidden_layer")(x)



    outputs = keras.layers.Dense(1, name="outputs")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Baseline_dense")


def dense_rnn_model(sequence_length):
    """
    Baseline Model RNN for RUL prediction

    Returns
    -------
    model: tensorflow model
        Tenssorflow model.

    """
    inputs = keras.Input(shape=(sequence_length, 18), name="inputs")
    x = keras.layers.LSTM(64, return_sequences=False, name="First_LSTM_layer")(inputs)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(1, name="predictions_rnn")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Baseline_rnn")




def SFE(input_shape=500, unit_custom=500):
    """
    Build a model with a custom layer, in this case a layer that extract the maximum value

    Parameters
    ----------
    input_shape : INT
        Input shape value. Usually equal to the window value.
    unit : INT
        Number of neuron per layer.
    unit_custom. INT
        Nu;ber of neurons for the modular modules

    Returns
    -------
    Keras model
        return a Keras model.

    """
    inputs = keras.Input(shape=(input_shape,), name="digits")

    # Time series
    cf = layers.CressFactorLayer(unit_custom)(inputs)
    pp = layers.PeakToPeak(unit_custom)(inputs)
    sf = layers.ShapeFactor(unit_custom)(inputs)
    impulse_f = layers.ImpulseFactor(unit_custom)(inputs)
    clearance_factor = layers.ClearanceFactor(unit_custom)(inputs)

    maximum = layers.SuperMaximum(unit_custom)(inputs)
    minimum = layers.SuperMinimum(unit_custom)(inputs)
    mean = layers.SuperMean(unit_custom)(inputs)
    var = layers.SuperVariance(unit_custom)(inputs)
    std = layers.SuperStd(unit_custom)(inputs)
    rms = layers.SuperRms(unit_custom)(inputs)
    skew = layers.SuperSkewness(unit_custom)(inputs)
    kurt = layers.SuperKurtosis(unit_custom)(inputs)

    # frequency
    fft = layers.SuperFft(unit_custom)(inputs)
    # magnitude = layers.SuperAmplitude(unit_custom)(fft)
    pmm = layers.PmmLayer(unit_custom)(fft)

    power_spectrum = layers.PowerSpectrumLayer(unit_custom)(fft)
    frequency_max = layers.SuperMaximum(unit_custom)(power_spectrum)
    frequency_mean = layers.SuperMean(unit_custom)(power_spectrum)
    frequency_var = layers.SuperVariance(unit_custom)(power_spectrum)
    frequency_sum = layers.SuperSum(unit_custom)(power_spectrum)

    x = keras.layers.concatenate(
        [cf, pp, sf, impulse_f, clearance_factor, maximum, minimum, mean, var, std, rms, skew,
         kurt, pmm, frequency_max, frequency_mean, frequency_var, frequency_sum], axis=1)

    x = keras.layers.Dense(64, activation="relu", name="first_hidden_layer")(x)
    x = keras.layers.Dense(64, activation="relu", name="second_hidden_layer")(x)

    outputs = keras.layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def model_custom_block_stat(input_shape=500,unit_custom=500,use_bias=False):
    """
    Build a model with a custom layer, in this case a layer that extract the maximum value

    Parameters
    ----------
    input_shape : INT
        Input shape value. Usually equal to the window value.
    unit : INT
        Number of neuron per layer.
    unit_custom. INT
        Nu;ber of neurons for the modular modules

    Returns
    -------
    Keras model
        return a Keras model.

    """
    inputs = keras.Input(shape=(input_shape,), name="digits")

    # Time series

    stat_feature = layers.StatisticalExtraction(unit_custom,use_bias)(inputs)
    freq_feature = layers.FrequencyExtraction(unit_custom,use_bias)(inputs)
    vibration_feature = layers.VibrationFeature(unit_custom,use_bias)(inputs)


    x = keras.layers.concatenate([stat_feature, freq_feature, vibration_feature], axis=-1)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)
