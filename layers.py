from tensorflow import keras
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
initializer_single_layer = "ones"


# kamal features
class PeakToPeak(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(PeakToPeak, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)

        # peak amplitude
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # minimum amplitude
        minimum = tf.math.reduce_min(inputs_weight, axis=1, keepdims=True)
        # distance
        peak_to_peak = tf.math.abs(maximum - minimum)
        return peak_to_peak


class CressFactorLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CressFactorLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)

        # peak amplitude
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # rms
        square = tf.math.square(inputs_weight)
        mean = tf.math.reduce_mean(square, axis=1, keepdims=True)
        rms = tf.math.sqrt(mean)
        crest_factor = maximum / rms
        return crest_factor


class ShapeFactor(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ShapeFactor, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)

        # rms
        square = tf.math.square(inputs_weight)
        mean = tf.math.reduce_mean(square, axis=1, keepdims=True)
        rms = tf.math.sqrt(mean)

        # absolute mean
        absolute = tf.math.abs(inputs_weight)
        mean = tf.math.reduce_mean(absolute, axis=1, keepdims=True)
        shape_factor = rms / mean
        return shape_factor


class ImpulseFactor(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ImpulseFactor, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        # peak amplitude
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # absolute mean
        absolute = tf.math.abs(inputs_weight)
        mean = tf.math.reduce_mean(absolute, axis=1, keepdims=True)
        impulse_factor = maximum / mean
        return impulse_factor


class ClearanceFactor(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        # peak amplitude
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # absolute mean
        absolute = tf.math.abs(inputs_weight)
        square_root = tf.math.sqrt(absolute)
        mean = tf.math.reduce_mean(square_root, axis=1, keepdims=True)
        clearance_factor = maximum / mean
        return clearance_factor


# statistical layer
class SuperMaximum(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperMaximum, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weighted = tf.multiply(inputs, self.w)
        maximum = tf.math.reduce_max(inputs_weighted, axis=1, keepdims=True)

        return maximum


class SuperMinimum(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperMinimum, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        minimum = tf.math.reduce_min(inputs_weight, axis=1, keepdims=True)
        return minimum


class Absolute(keras.layers.Layer):
    def __init__(self,
                 units,
                 activity_regularizer=None,
                 **kwargs,
                 ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = units

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer="glorot_uniform",
            trainable=True,
        )

        self.bias = self.add_weight(
            "bias",
            shape=[
                self.units,
            ],
            trainable=True,
        )

    def call(self, inputs):
        return tf.abs(tf.matmul(a=inputs, b=self.kernel) + self.bias)


class SuperMean(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperMean, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        mean = tf.math.reduce_mean(inputs_weight, axis=1, keepdims=True)
        return mean


class SuperVariance(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperVariance, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        variance = tf.math.reduce_variance(inputs_weight, axis=1, keepdims=True)
        return variance


class SuperMaxMin(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperMaxMin, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        minimum = tf.math.reduce_min(inputs_weight, axis=1, keepdims=True)
        return (maximum - minimum)


class SuperStd(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperStd, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        std = tf.math.reduce_std(inputs_weight, axis=1, keepdims=True)
        return std


class SuperRms(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperRms, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        square = tf.math.square(inputs_weight)
        mean = tf.math.reduce_mean(square, axis=1, keepdims=True)
        rms = tf.math.sqrt(mean)
        return rms


class SuperSkewness(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperSkewness, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):

        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def skewnessClass(self, inputs, axis=0):

        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        m2 = self._momentClass(inputs, 2, axis, mean=mean)
        m3 = self._momentClass(inputs, 3, axis, mean=mean)

        testComparaison = (tf.experimental.numpy.finfo(m2.dtype).resolution * mean)
        testComparaison = tf.cast(testComparaison, dtype='float32')
        zero = (m2 <= (testComparaison) ** 2)
        vals = tf.experimental.numpy.where(zero, np.nan, m3 / m2 ** 1.5)

        return vals

    def _momentClass(self, inputs, moment, axis, *, mean=None):

        # moment of empty array is the same regardless of order

        if moment == 0 or moment == 1:
            # By definition the zeroth moment about the mean is 1, and the first
            # moment is 0.
            tensorShape = tf.shape(inputs)
            a_vecs = tf.unstack(tensorShape, axis)
            del a_vecs[axis]


        else:
            # Exponentiation by squares: form exponent sequence
            n_list = [moment]
            current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = tf.math.reduce_mean(inputs, axis, keepdims=True) if mean is None else mean
        meanFloat = tf.cast(mean, dtype='float32')
        aFloat = tf.cast(inputs, dtype='float32')
        a_zero_mean = aFloat - meanFloat

        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = tf.math.square(a_zero_mean)

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = tf.math.square(s)
            if n % 2:
                s = tf.math.multiply(s, a_zero_mean)

        return tf.math.reduce_mean(s, 1, keepdims=True)

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        skew = self.skewnessClass(inputs_weight)

        return skew


class SuperKurtosis(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperKurtosis, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):

        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def kurtosisClass(self, inputs, axis=0):

        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        m2 = self._momentClass(inputs, 2, axis, mean=mean)
        m4 = self._momentClass(inputs, 4, axis, mean=mean)

        testComparaison = (tf.experimental.numpy.finfo(m2.dtype).resolution * mean)
        testComparaison = tf.cast(testComparaison, dtype='float32')
        zero = (m2 <= (testComparaison) ** 2)
        vals = tf.experimental.numpy.where(zero, np.nan, m4 / m2 ** 2.0)

        return vals - 3

    def _momentClass(self, inputs, moment, axis, *, mean=None):

        # moment of empty array is the same regardless of order

        if moment == 0 or moment == 1:
            # By definition the zeroth moment about the mean is 1, and the first
            # moment is 0.
            tensorShape = tf.shape(inputs)
            a_vecs = tf.unstack(tensorShape, axis)
            del a_vecs[axis]


        else:
            # Exponentiation by squares: form exponent sequence
            n_list = [moment]
            current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = tf.math.reduce_mean(inputs, axis, keepdims=True) if mean is None else mean
        meanFloat = tf.cast(mean, dtype='float32')
        aFloat = tf.cast(inputs, dtype='float32')
        a_zero_mean = aFloat - meanFloat

        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = tf.math.square(a_zero_mean)

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = tf.math.square(s)
            if n % 2:
                s = tf.math.multiply(s, a_zero_mean)

        return tf.math.reduce_mean(s, 1, keepdims=True)

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        kurt = self.kurtosisClass(inputs_weight)
        return kurt


class SuperSum(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SuperSum, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        somme = tf.math.reduce_sum(inputs_weight, axis=1, keepdims=True)

        return somme


# frequency layers

class SuperFft(keras.layers.Layer):
    """
    Directly return the FFT
    """

    def __init__(self, units, **kwargs):
        super(SuperFft, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        inputs = tf.cast(inputs_weight, tf.complex64, name=None)
        fft = tf.signal.fft(inputs)
        magnitude = tf.abs(fft)
        return magnitude


class SuperAmplitude(keras.layers.Layer):
    """
    Return the magnitude. Have to be put after FFT
    """

    def __init__(self, units, **kwargs):
        super(SuperAmplitude, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, tf.cast(self.w, tf.complex64))
        magnitude = tf.math.abs(inputs_weight)
        return magnitude


class PmmLayer(keras.layers.Layer):
    """
    Have to be put after a magnitude layer
    Calculate the ratio max(frequency)/mean(frequency)

    """

    def __init__(self, units, **kwargs):
        super(PmmLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        # peak amplitude
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # absolute mean
        mean = tf.math.reduce_mean(inputs_weight, axis=1, keepdims=True)
        PMM = maximum / mean
        return PMM


class PowerSpectrumLayer(keras.layers.Layer):
    """
    Have to be put after a Amplitude layer
    Power Spectrum

    """

    def __init__(self, units, **kwargs):
        super(PowerSpectrumLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_single_layer,
            trainable=True,
        )

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        power_spectrum = tf.math.square(inputs_weight) / tf.cast(tf.size(inputs), tf.float32)

        return power_spectrum


initializer_block = 'ones'
initializer_bias_block = "random_normal"


class StatisticalExtraction(keras.layers.Layer):
    """
    Extract all basic statistics

    """

    def __init__(self, units, use_bias, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.bias_initializer = "ones"

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_block,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(6,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        minimum = tf.math.reduce_min(inputs_weight, axis=1, keepdims=True)
        mean = tf.math.reduce_mean(inputs_weight, axis=1, keepdims=True)
        variance = tf.math.reduce_variance(inputs_weight, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs_weight, axis=1, keepdims=True)
        square = tf.math.square(inputs_weight)
        mean_square = tf.math.reduce_mean(square, axis=1, keepdims=True)
        rms = tf.math.sqrt(mean_square)
        outputs = tf.concat([maximum, minimum, mean, variance, std, rms], axis=1)

        if self.use_bias:
            outputs = tf.multiply(outputs, self.bias)

        return outputs


class FrequencyExtraction(keras.layers.Layer):
    """
    Extract all basic statistics

    """

    def __init__(self, units, use_bias, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.bias_initializer = "ones"

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_block,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(5,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        fft = tf.signal.fft(tf.cast(inputs_weight, tf.complex64, name=None))
        magnitude = tf.math.abs(fft)
        power_spectrum = tf.math.square(magnitude) / tf.cast(tf.size(inputs), tf.float32)
        maximum_magnitude = tf.math.reduce_max(magnitude, axis=1, keepdims=True)
        mean_magnitude = tf.math.reduce_mean(magnitude, axis=1, keepdims=True)
        PMM = maximum_magnitude / mean_magnitude
        maximum = tf.math.reduce_max(power_spectrum, axis=1, keepdims=True)
        mean = tf.math.reduce_mean(power_spectrum, axis=1, keepdims=True)
        somme = tf.math.reduce_sum(power_spectrum, axis=1, keepdims=True)
        variance = tf.math.reduce_variance(power_spectrum, axis=1, keepdims=True)
        outputs = tf.concat([PMM, maximum, mean, variance, somme], axis=1)
        if self.use_bias:
            outputs = tf.multiply(outputs, self.bias)
        return outputs


class VibrationFeature(keras.layers.Layer):
    """
    Extract all basic statistics

    """

    def __init__(self, units, use_bias, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.bias_initializer = "ones"

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(self.units,),
            initializer=initializer_block,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(7,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None

    def kurtosisClass(self, inputs, axis=0):

        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        m2 = self._momentClass(inputs, 2, axis, mean=mean)
        m4 = self._momentClass(inputs, 4, axis, mean=mean)

        testComparaison = (tf.experimental.numpy.finfo(m2.dtype).resolution * mean)
        testComparaison = tf.cast(testComparaison, dtype='float32')
        zero = (m2 <= (testComparaison) ** 2)
        vals = tf.experimental.numpy.where(zero, np.nan, m4 / m2 ** 2.0)

        return vals - 3

    def skewnessClass(self, inputs, axis=0):

        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        m2 = self._momentClass(inputs, 2, axis, mean=mean)
        m3 = self._momentClass(inputs, 3, axis, mean=mean)

        testComparaison = (tf.experimental.numpy.finfo(m2.dtype).resolution * mean)
        testComparaison = tf.cast(testComparaison, dtype='float32')
        zero = (m2 <= (testComparaison) ** 2)
        vals = tf.experimental.numpy.where(zero, np.nan, m3 / m2 ** 1.5)

        return vals

    def _momentClass(self, inputs, moment, axis, *, mean=None):

        # moment of empty array is the same regardless of order

        if moment == 0 or moment == 1:
            # By definition the zeroth moment about the mean is 1, and the first
            # moment is 0.
            tensorShape = tf.shape(inputs)
            a_vecs = tf.unstack(tensorShape, axis)
            del a_vecs[axis]


        else:
            # Exponentiation by squares: form exponent sequence
            n_list = [moment]
            current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = tf.math.reduce_mean(inputs, axis, keepdims=True) if mean is None else mean
        meanFloat = tf.cast(mean, dtype='float32')
        aFloat = tf.cast(inputs, dtype='float32')
        a_zero_mean = aFloat - meanFloat

        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = tf.math.square(a_zero_mean)

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = tf.math.square(s)
            if n % 2:
                s = tf.math.multiply(s, a_zero_mean)

        return tf.math.reduce_mean(s, 1, keepdims=True)

    def call(self, inputs):
        inputs_weight = tf.multiply(inputs, self.w)
        maximum = tf.math.reduce_max(inputs_weight, axis=1, keepdims=True)
        # minimum amplitude
        minimum = tf.math.reduce_min(inputs_weight, axis=1, keepdims=True)
        # distance
        peak_to_peak = tf.math.abs(maximum - minimum)

        square = tf.math.square(inputs_weight)
        mean = tf.math.reduce_mean(square, axis=1, keepdims=True)
        rms = tf.math.sqrt(mean)
        crest_factor = maximum / rms

        absolute = tf.math.abs(inputs_weight)
        mean_abs = tf.math.reduce_mean(absolute, axis=1, keepdims=True)
        shape_factor = rms / mean_abs

        impulse_factor = maximum / mean

        square_root_abs = tf.math.sqrt(absolute)
        mean_sqrt_abs = tf.math.reduce_mean(square_root_abs, axis=1, keepdims=True)
        clearance_factor = maximum / mean_sqrt_abs

        kurt = self.kurtosisClass(inputs_weight)
        skewness = self.skewnessClass(inputs_weight)
        outputs = tf.concat(
            [peak_to_peak, crest_factor, shape_factor, impulse_factor, clearance_factor, kurt, skewness],
            axis=1)

        if self.use_bias:
            outputs = tf.multiply(outputs, self.bias)
        return outputs
