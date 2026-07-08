import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import pennylane as qml
import time

from keras import initializers
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K
K.set_image_data_format('channels_last')

print(len(tf.config.list_physical_devices('GPU')))


data = pd.read_excel('ENB2012_data.xlsx')

# Column names for convenience
data.columns = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 'Overall_Height', 
    'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load'
]

# Inputs (Features)
X = data.iloc[:, :-2].values  # All columns except the last two

# Outputs (Targets)
y = data.iloc[:, -2:].values  # Last two columns (Heating Load and Cooling Load)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Min-Max Scaling for input features to [0, 1]
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)  # Scale the training set
X_test_scaled = scaler_X.transform(X_test)        # Apply the same scaling to the test set

# Min-Max Scaling for target variables to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train)  # Scale target variables for training
y_test_scaled = scaler_y.transform(y_test)        # Apply the same scaling to the test target variables

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

print(X_train_scaled.min(),X_train_scaled.max())
print(y_train_scaled.min(),y_train_scaled.max())


num_qubits=16

dev = qml.device("lightning.gpu", wires=num_qubits)

tf.keras.backend.set_floatx('float64')

@qml.qnode(dev, interface="tf", diff_method="adjoint")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
    #return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
    
    # Define the groups of wires
    wire_groups = [range(0, 2), range(2,4), range(4,6), range(6,8), range(8,10), range(10,12), range(12,14), range(14,16)]

   # Measure the expectation value for each group
    expectation_values = []
    for group in wire_groups:
        # Create a wire_map for the current group
        wire_map = {wire: idx for idx, wire in enumerate(group)}

        # Create the Pauli string
        pauli_string = 'Z' * len(group)

        # Convert the string to a Pauli word with the corresponding wire_map
        pauli_word = qml.pauli.string_to_pauli_word(pauli_string, wire_map)

        # Append the expectation value to the list
        expectation_values.append(qml.expval(pauli_word))

    return expectation_values


n_layers = 6
weight_shapes = {"weights": (n_layers, num_qubits)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=8)


given_seed = 2

input_final = Input(shape=(X_train.shape[1:]))

Z = Dense(64, name='fc0',
          kernel_initializer=initializers.he_uniform(seed=given_seed), bias_initializer='zeros')(input_final)

Z = Activation('relu')(Z)

Z = Dense(16, name='fc1',
          kernel_initializer=initializers.he_uniform(seed=given_seed), bias_initializer='zeros')(Z)

Z = Activation('relu')(Z)

Z = qlayer(Z)

#Z = Dense(8, name='fc2',
#          kernel_initializer=initializers.he_uniform(seed=given_seed), bias_initializer='zeros')(Z)

#Z = Activation('relu')(Z)

output = Dense(2, name='fc4',
          kernel_initializer=initializers.he_uniform(seed=given_seed), bias_initializer='zeros')(Z)


print(output)


sup_net = Model(inputs=input_final, outputs=output)


class my_loss_fun(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        mse_res1 = tf.reduce_mean(tf.square(y_pred-y_true))
        return tf.math.sqrt(mse_res1)


adam =tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

sup_net.compile(loss=my_loss_fun(), optimizer = adam)


mc = ModelCheckpoint('/scratch/users/divakarv/sup_8inp_2out_hyb_1_2_gpu.h5', monitor='val_loss', mode='min', verbose=1, save_weights_only=True)


# Start the training
start = time.time()

Batch_size=32
history = sup_net.fit(X_train_scaled, y_train_scaled, validation_split=0.2, 
                         epochs=200, batch_size=Batch_size, callbacks=[mc],
                        verbose = 1)

end = time.time()
print(end - start)


# load the saved model
sup_net.load_weights('/scratch/users/divakarv/sup_8inp_2out_hyb_1_2_gpu.h5')

sup_net.save_weights('sup_8inp_2out_hyb_1_2_gpu.h5')


# Testing and evaluating the model
preds = sup_net.evaluate(X_test_scaled, y_test_scaled)
print()
print ("Loss = " + str(preds))

preds = sup_net.evaluate(X_train_scaled, y_train_scaled, batch_size=Batch_size)
print()
print ("Loss = " + str(preds))


print('Training Loss:') 
print(history.history['loss'])

print('Validation Loss:')
print(history.history['val_loss'])

np.save('train_loss_8inp_2out_hyb_1_2_gpu.npy', history.history['loss'])
np.save('val_loss_8inp_2out_hyb_1_2_gpu.npy', history.history['val_loss'])


# Predicting on the train set
y_pred_train_scaled = sup_net.predict(X_train_scaled)

# Transform predictions back to original scale
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)

# Inverse transform the scaled targets (to compare the actual values with the predictions)
y_train_original = scaler_y.inverse_transform(y_train_scaled)

# Display results
for i in range(5):
    print(f"Actual: {y_train[i]}, Predicted: {y_pred_train[i]}")
    

np.save('y_pred_train_8inp_2out_hyb_1_2_gpu.npy',y_pred_train)


# Predicting on the test set
y_pred_test_scaled = sup_net.predict(X_test_scaled)

# Transform predictions back to original scale
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)

# Inverse transform the scaled targets (to compare the actual values with the predictions)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Display results
for i in range(5):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred_test[i]}")
    
np.save('y_pred_test_8inp_2out_hyb_1_2_gpu.npy',y_pred_test)


print(f"RMSE: {np.sqrt(np.mean(np.square(y_train-y_pred_train)))}, MSE: {np.mean(np.square(y_train-y_pred_train))}, MAE: {np.mean(np.abs(y_train-y_pred_train))}")

print(f"RMSE: {np.sqrt(np.mean(np.square(y_test-y_pred_test)))}, MSE: {np.mean(np.square(y_test-y_pred_test))}, MAE: {np.mean(np.abs(y_test-y_pred_test))}")


# Calculate residuals
residuals_test = y_test - y_pred_test

np.save('residuals_test_8inp_2out_hyb_1_2_gpu.npy',residuals_test)


print('Time taken:', (end - start)/3600)