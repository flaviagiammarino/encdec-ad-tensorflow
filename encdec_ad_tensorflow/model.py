import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from encdec_ad_tensorflow.modules import EncoderDecoder

class EncDecAD():
    
    def __init__(self, m, L=10, c=100, beta=0.1, num=100):
    
        '''
        Implementation of time series anomaly detection model introduced in Malhotra, P., Ramakrishnan, A.,
        Anand, G., Vig, L., Agarwal, P. and Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor
        anomaly detection.
        '''
        
        self.m = m
        self.c = c
        self.L = L
        self.beta = beta
        self.num = num
    
    def fit(self,
            xn,
            xa,
            learning_rate=0.001,
            batch_size=32,
            max_epochs=100,
            early_stopping_start_epoch=100,
            early_stopping_patience=10,
            verbose=1):
        
        '''
        Train the model.
        '''
       
        # Split the time series into sequences.
        x_sn = time_series_to_sequences(xn, self.L)
        x_va = time_series_to_sequences(xa, self.L)
        
        # Split the normal sequences into a training set and two validation sets.
        x_sn, x_vn = train_test_split(x_sn, test_size=0.5, random_state=42)
        x_vn1, x_vn2 = train_test_split(x_vn, test_size=0.5, random_state=42)
        
        # Calculate the scaling parameters.
        x_min = np.nanmin(x_sn, axis=0, keepdims=True)
        x_max = np.nanmax(x_sn, axis=0, keepdims=True)
        
        # Scale the sequences.
        x_sn = (x_sn - x_min) / (x_max - x_min)
        x_vn1 = (x_vn1 - x_min) / (x_max - x_min)
        x_vn2 = (x_vn2 - x_min) / (x_max - x_min)
        x_va = (x_va - x_min) / (x_max - x_min)
        
        # Build the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_sn, reverse_sequences(x_sn)))
        train_dataset = train_dataset.cache().shuffle(len(x_sn)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Build the validation dataset.
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_vn1, reverse_sequences(x_vn1)))
        valid_dataset = valid_dataset.batch(batch_size)
        
        # Build the model.
        model = EncoderDecoder(L=self.L, m=self.m, c=self.c)

        # Train the model.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta=0,
            patience=early_stopping_patience,
            start_from_epoch=early_stopping_start_epoch,
            restore_best_weights=True
        )

        model.fit(
            train_dataset,
            epochs=max_epochs,
            validation_data=valid_dataset,
            callbacks=[callback],
            verbose=verbose
        )
        
        # Generate the reconstructions.
        r_vn1 = model(x_vn1, training=False)
        r_vn2 = model(x_vn2, training=False)
        r_va = model(x_va, training=False)

        # Calculate the reconstruction errors.
        e_vn1 = tf.math.abs(x_vn1 - r_vn1)
        e_vn2 = tf.math.abs(x_vn2 - r_vn2)
        e_va = tf.math.abs(x_va - r_va)

        # Calculate the mean vector and covariance matrix.
        mu, sigma = get_mu_and_sigma(x=sequences_to_time_series(e_vn1))

        # Calculate the anomaly scores.
        a_vn2 = get_anomaly_scores(e_vn2, mu, sigma)
        a_va = get_anomaly_scores(e_va, mu, sigma)

        # Find the best threshold.
        tau = get_anomaly_threshold(
            an=a_vn2,
            aa=a_va,
            beta=self.beta,
            num=self.num
        )
        
        self.model = model
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.x_min = x_min
        self.x_max = x_max
    
    def predict(self, x):
        
        # Split the time series into sequences
        x = time_series_to_sequences(x, self.L)
        
        # Scale the sequences.
        x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # Generate the reconstructions.
        r = self.model(x, training=False)
        
        # Calculate the reconstruction errors.
        e = tf.math.abs(x - r)

        # Calculate the anomaly scores.
        a = get_anomaly_scores(e, self.mu, self.sigma)

        # Calculate the anomaly labels.
        y = get_anomaly_labels(a, self.tau)
        
        # Transform the reconstructions back to the original scale.
        r = self.x_min + (self.x_max - self.x_min) * r

        # Transform the sequences back to time series.
        r = sequences_to_time_series(r)
        a = sequences_to_time_series(a)
        y = np.concatenate([np.repeat(y[i], repeats=self.L, axis=0) for i in range(y.shape[0])], axis=0)
        
        return r, a, y


def time_series_to_sequences(x, L):
    # Split the time series into sequences.
    if len(x) % L == 0:
        return np.array([x[i - L: i] for i in range(L, L * len(x) // L + L, L)])
    else:
        return np.array([x[i - L: i] for i in range(L, L * len(x) // L, L)])


def sequences_to_time_series(x):
    # Transform the sequences back to time series.
    return np.concatenate([x[i] for i in range(len(x))])


def reverse_sequences(x):
    # Reverse the order of the sequences.
    return np.concatenate([np.expand_dims(np.flip(x[i], axis=0), axis=0) for i in range(len(x))], axis=0)


def get_mu_and_sigma(x):
    # Calculate the mean vector and covariance matrix.
    mu = tf.reduce_mean(x, axis=0, keepdims=True)
    sigma = tf.divide(tf.matmul(x - tf.reduce_mean(x, axis=0), x - tf.reduce_mean(x, axis=0), transpose_a=True), x.shape[0] - 1)
    return mu, sigma


def get_anomaly_scores(x, mu, sigma):
    # Calculate the anomaly scores.
    fn = lambda x: tf.squeeze(tf.matmul(tf.matmul(x - mu, tf.linalg.inv(sigma)), x - mu, transpose_b=True))
    return tf.transpose([tf.map_fn(elems=x[:, i, :], fn=fn) for i in range(x.shape[1])], perm=(1, 0))


def get_anomaly_labels(a, tau):
    # Derive the anomaly labels.
    return tf.where(tf.reduce_any(a > tau, axis=1, keepdims=True), 1., 0.)
    

def get_anomaly_threshold(an, aa, beta, num):
    
    # Concatenate the anomaly scores.
    a = tf.concat([an, aa], axis=0)
    
    # Define the list of thresholds.
    taus = tf.linspace(start=tf.reduce_min(a), stop=tf.reduce_max(a), num=num)

    # Instantiate the F-beta scorer.
    scorer = tf.metrics.FBetaScore(beta=beta)
    
    # Create a tensor for storing the F-beta scores of the thresholds.
    scores = tf.TensorArray(
        element_shape=(),
        size=num,
        dynamic_size=False,
        dtype=tf.float32,
        clear_after_read=False
    )
    
    # Derive the ground-truth anomaly labels.
    y_true = tf.concat([tf.zeros((an.shape[0], 1)), tf.ones((aa.shape[0], 1))], axis=0)
    
    # Loop across the thresholds.
    for i in tf.range(start=0, limit=num, delta=1):
        
        # Derive the predicted anomaly labels.
        y_pred = get_anomaly_labels(a, tau=taus[i])
        
        # Calculate and save the F-beta score.
        scores = scores.write(index=i, value=tf.squeeze(scorer(y_true, y_pred)))
    
    # Return the threshold with the maximum F-beta score.
    return taus[tf.argmax(scores.stack())]
