The data can be found by following this link: https://drive.google.com/drive/folders/1cY_wlC8wrFxlVg1xrdfN-FrSU0Ob64Sg?usp=drive_link 

# Neuron units for feature extraction
This github post introduces a novel approach for extracting features for a RUL prediction problem by using neuron units that extract the features inside the model. The core idea behind this approach is to optimize the feature extraction by the models, as the inputs are first multiplied by the weights, and then the features are extracted.

## Overview

The neuron units are designed to be incoporated in a Multi Layer Perceptron (MLP) as we only tested their implementation and impact in this type of model. The weights of the neuron units are trainable, and we use the initializer "ones". The neuron units are placed after the input layer, as they need to extract the features before feeding them to the dense layers. We do have thee types of neuron units for features extraction, one that only extract one feature or do one operation, one that extract multiple features and one that extract multiple features but multiply the features extracted by a bias (which revealed to be counter productive as the dense layers were already doing this).  We do use a Leave One Out to test the different models, by recreating a model for every training and testing set. One bearing is then used as the test set while all the other bearings and their corresponding RUL are used as the training set. We then calculate the Root Mean Squared Error (RMSE), the Mean Absolute Error (MAE), and the alpha-lambda

We see some improvements for the RUL prediction, but the most notable one is on the first bearing of the training set. Here are some potentials that can be more explored in future research:
- **Interpretability**: Even though we didn't explain the interpretability aspect of these neuron units, we believe that by adding neuron units that have a clear purpose can help enhance the overall interpretability of the model, as each output of the neuron units is more comprehensive.
- **Modularity**: By having these neuron units, we can have a modular model that can be adapted to the case studied while also changing the model architecture.

## Known Limitations

We do observe that for some bearings in the training test, the baseline outperforms the modular approach. However, despite different runs, the baseline never outperforms the modular approaches on the first bearing (which can be due to the size of the array of the testing set being almost equal to 50% of the size of the array of all the different bearings used as the training set). Regarding the runs, we can have the baseline model have a lower MAE than the modular models. Overall, the modular approaches tend to have a lower RMSE and the highest alpha-lambda. The effectiveness of the neuron units seems to work overall, but deeper research is needed to fully understand how they improve, in some cases, the prediction, while in others they worsen it.

## Future Research

Despite the limitations, the neuron units represent an intriguing approach on how to implement knowledge inside of a neural network, and mostly with the results of the first bearing where the difference between the baseline and the modular models are big. Future work include:

- **Theoretical Analysis**: Conducting theoretical analysis to better understand why this modular approach works on some bearings and why it is working on other bearings. A better understanding of how they work can lead to an improved architecture of the neuron units, of the neural network, or of the of the training strategy.
- **Task-specific Optimization**: Investigating the modification needed to enhance the output given by the neural network for a specific case study.
- **Interpretability**: Exploring the effect of these modular neuron units on the interpretability of the model, enabling a better understanding of the output produced by the model during training, should also be explored.


## Conclusion

Whil the current neuron units, have limitations, I do believe that they can be optimized by changing the knowledge inside of them, their architecture or exploring alternative integration strategies within the overall model, it is possible to enhance their performance.
This GitHub repository serves as a starting point for further exploration, experimentation, and advancement in this direction.
