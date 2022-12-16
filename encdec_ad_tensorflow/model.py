import numpy as np
import pandas as pd
import tensorflow as tf

from encdec_ad_tensorflow.modules import EncoderDecoder

class EncDecAD():
    
    def __init__(self, x, units, timesteps):
    
        '''
        Implementation of multivariate time series anomaly detection model introduced in Malhotra, P., Ramakrishnan, A.,
        Anand, G., Vig, L., Agarwal, P. and Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, features) where samples is the length of the time series and features
            is the number of time series.

        units: int.
            Number of hidden units of the LSTM layers.
            
        timesteps: int.
            Number of time steps.
        '''
        
        self.x = x
        self.x_min = np.min(x, axis=0)
        self.x_max = np.max(x, axis=0)
        self.samples = x.shape[0]
        self.features = x.shape[1]
        self.units = units
        self.timesteps = timesteps

    def fit(self, learning_rate=0.001, batch_size=32, epochs=100, verbose=True):
        
        '''
        Train the model.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
    
        # Scale the time series.
        x = (self.x - self.x_min) / (self.x_max - self.x_min)

        # Generate the input sequences.
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=tf.cast(x, tf.float32),
            targets=None,
            sequence_length=self.timesteps,
            sequence_stride=self.timesteps,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Build the model.
        inputs = tf.keras.layers.Input(shape=(self.timesteps, self.features))
        outputs = EncoderDecoder(units=self.units)(inputs)
        model = tf.keras.models.Model(inputs, outputs)
        
        # Define the training loop.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                
                # Calculate the loss.
                output = model(data, training=True)
                loss = tf.reduce_mean(tf.reduce_sum((data - output) ** 2, axis=-1))
        
            # Calculate the gradient.
            gradient = tape.gradient(loss, model.trainable_variables)
    
            # Update the weights.
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    
            return loss

        # Train the model.
        for epoch in range(epochs):
            for data in dataset:
                loss = train_step(data)
            if verbose:
                print('epoch: {}, loss: {:,.6f}'.format(1 + epoch, loss))

        # Save the model.
        self.model = model

    def predict(self, x):
    
        '''
        Reconstruct the time series and score the anomalies.

        Parameters:
        __________________________________
        x: np.array.
            Actual time series, array with shape (samples, features) where samples is the length
            of the time series and features is the number of time series.

        Returns:
        __________________________________
        x_hat: np.array.
            Reconstructed time series, array with shape (samples, features) where samples is the
            length of the time series and features is the number of time series.
    
        scores: np.array.
            Anomaly scores, array with shape (samples,) where samples is the length of the time
            series.
        '''
    
        if x.shape[1] != self.features:
            raise ValueError(f'Expected {self.features} features, found {x.shape[1]}.')
    
        else:

            # Generate the reconstructions.
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=tf.cast((x - self.x_min) / (self.x_max - self.x_min), tf.float32),
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=self.timesteps,
                batch_size=1,
                shuffle=False
            )
        
            x_hat = np.concatenate([self.model(data, training=False).numpy() for data in dataset], axis=0)
            x_hat = np.concatenate([x_hat[i, :, :] for i in range(x_hat.shape[0])], axis=0)
            x_hat = self.x_min + (self.x_max - self.x_min) * x_hat

            # Calculate the anomaly scores.
            errors = np.abs(x - x_hat)
            mu = np.mean(errors, axis=0)
            sigma = np.cov(errors, rowvar=False)
            scores = np.array([np.dot(np.dot((errors[i, :] - mu).T, np.linalg.inv(sigma)), (errors[i, :] - mu)) for i in range(errors.shape[0])])

            return x_hat, scores
