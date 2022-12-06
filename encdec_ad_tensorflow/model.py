import numpy as np
import pandas as pd
import tensorflow as tf

from encdec_ad_tensorflow.modules import EncoderDecoder
from encdec_ad_tensorflow.plots import plot

class EncDecAD():
    
    def __init__(self, x, units, timesteps):
    
        '''
        Implementation of multivariate time series anomaly detection model introduced in Malhotra, P., Ramakrishnan, A.,
        Anand, G., Vig, L., Agarwal, P. and Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection.
        arXiv preprint arXiv:1607.00148. https://arxiv.org/abs/1607.00148.

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
            Time series, array with shape (samples, features) where samples is the length of the time series
            and features is the number of time series.

        Returns:
        __________________________________
        reconstructions: pd.DataFrame.
            Data frame with reconstructed time series.
        
        scores: pd.DataFrame.
            Data frame with anomaly scores.
        '''
    
        if x.shape[1] != self.features:
            raise ValueError(f'Expected {self.features} features, found {x.shape[1]}.')
    
        else:
        
            # Save the time series.
            self.actual = pd.DataFrame(x)
        
            # Scale the time series.
            x = (x - self.x_min) / (self.x_max - self.x_min)
        
            # Generate the input sequences.
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=tf.cast(x, tf.float32),
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=self.timesteps,
                batch_size=1,
                shuffle=False
            )
        
            # Generate the reconstructions.
            outputs = tf.concat([self.model(data, training=False) for data in dataset], axis=0).numpy()
            reconstructions = pd.DataFrame()
            for i in range(outputs.shape[0]):
                reconstructions = pd.concat([reconstructions, pd.DataFrame(outputs[i, :, :])], axis=0, ignore_index=True)
            reconstructions = self.x_min + (self.x_max - self.x_min) * reconstructions

            # Calculate the anomaly scores.
            errors = (self.actual - reconstructions).abs()
            mu = errors.mean().values.reshape(-1, 1)
            sigma = errors.cov().values
            scores = pd.DataFrame()
            for i in range(errors.shape[0]):
                e = errors.iloc[i, :].values.reshape(-1, 1)
                scores = pd.concat([scores, pd.DataFrame(np.dot(np.dot((e - mu).T, np.linalg.inv(sigma)), (e - mu)))], axis=0, ignore_index=True)
        
            # Save the results.
            self.reconstructions = reconstructions
            self.scores = scores
            
            return reconstructions, scores

    def plot(self, quantile):
    
        '''
        Plot the results.
        
        Parameters:
        __________________________________
        quantile: float.
            Quantile of anomaly score used for identifying the anomalies.
            
        Returns:
        __________________________________
        go.Figure.
        '''
    
        return plot(x=self.actual.values, y=self.reconstructions.values, s=self.scores.values.flatten(), quantile=quantile)
    
