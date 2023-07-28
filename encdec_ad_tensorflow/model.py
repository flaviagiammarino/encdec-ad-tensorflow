import tensorflow as tf
from sklearn.metrics import fbeta_score

from encdec_ad_tensorflow.modules import EncoderDecoder

class EncDecAD():
    
    def __init__(self, m, L=10, c=100, beta=0.1, num=100):
    
        '''
        Implementation of multivariate time series anomaly detection model introduced in Malhotra, P., Ramakrishnan, A.,
        Anand, G., Vig, L., Agarwal, P. and Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection.
        '''
        
        self.m = m
        self.c = c
        self.L = L
        self.beta = float(beta)
        self.num = num
    
    def fit(self,
            xn,
            xa,
            ya,
            train_size=0.5,
            learning_rate=0.001,
            batch_size=32,
            max_epochs=100,
            early_stopping_start_epoch=10,
            early_stopping_patience=1,
            verbose=1):
        
        '''
        Train the model.
        '''
       
        # Process the normal time series.
        x_sn = tf.cast(xn[:int(train_size * len(xn))], dtype=tf.float32)
        x_vn1 = tf.cast(xn[int(train_size * len(xn)): - int(train_size * len(xn) / 2)], dtype=tf.float32)
        x_vn2 = tf.cast(xn[- int(train_size * len(xn) / 2):], dtype=tf.float32)
        y_vn2 = tf.zeros(len(x_vn2), dtype=tf.float32)
        
        # Process the anomalous time series.
        x_va = tf.cast(xa, dtype=tf.float32)
        y_va = tf.cast(ya, dtype=tf.float32)
        
        # Make sure that the length of the time series is a multiple of the sequence length.
        x_sn = x_sn[:self.L * (len(x_sn) // self.L)]
        x_vn1 = x_vn1[:self.L * (len(x_vn1) // self.L)]
        x_vn2 = x_vn2[:self.L * (len(x_vn2) // self.L)]
        y_vn2 = y_vn2[:self.L * (len(y_vn2) // self.L)]
        x_va = x_va[:self.L * (len(x_va) // self.L)]
        y_va = y_va[:self.L * (len(y_va) // self.L)]
        
        # Split the time series into sequences.
        xs_sn = time_series_to_sequences(x_sn, L=self.L)
        xs_vn1 = time_series_to_sequences(x_vn1, L=self.L)
        xs_vn2 = time_series_to_sequences(x_vn2, L=self.L)
        xs_va = time_series_to_sequences(x_va, L=self.L)

        # Calculate the scaling factors.
        x_min = tf.reduce_min(xs_sn, axis=0, keepdims=True)
        x_max = tf.reduce_max(xs_sn, axis=0, keepdims=True)

        # Scale the time series.
        xs_sn = (xs_sn - x_min) / (x_max - x_min)
        xs_vn1 = (xs_vn1 - x_min) / (x_max - x_min)
        xs_vn2 = (xs_vn2 - x_min) / (x_max - x_min)
        xs_va = (xs_va - x_min) / (x_max - x_min)
        
        # Build the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((xs_sn, reverse_sequences(xs_sn)))
        train_dataset = train_dataset.cache().shuffle(len(xs_sn)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Build the validation dataset.
        valid_dataset = tf.data.Dataset.from_tensor_slices((xs_vn1, reverse_sequences(xs_vn1)))
        valid_dataset = valid_dataset.batch(batch_size)
        
        # Build the model.
        model = EncoderDecoder(
            L=self.L,
            m=self.m,
            c=self.c
        )
        
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
        rs_vn1 = model(xs_vn1, training=False)
        rs_vn2 = model(xs_vn2, training=False)
        rs_va = model(xs_va, training=False)

        # Transform the reconstructions back to the original scale.
        rs_vn1 = x_min + (x_max - x_min) * rs_vn1
        rs_vn2 = x_min + (x_max - x_min) * rs_vn2
        rs_va = x_min + (x_max - x_min) * rs_va
        
        # Transform the reconstructions back to time series.
        r_vn1 = sequences_to_time_series(rs_vn1)
        r_vn2 = sequences_to_time_series(rs_vn2)
        r_va = sequences_to_time_series(rs_va)
        
        # Calculate the reconstruction errors.
        e_vn1 = tf.math.abs(x_vn1 - r_vn1)
        e_vn2 = tf.math.abs(x_vn2 - r_vn2)
        e_va = tf.math.abs(x_va - r_va)

        # Calculate the mean vector and covariance matrix.
        mu = tf.reduce_mean(e_vn1, axis=0, keepdims=True)
        sigma = tf.matmul(e_vn1 - mu, e_vn1 - mu, transpose_a=True) / (len(e_vn1) - 1)
        
        # Calculate the anomaly scores.
        a_va = get_anomaly_scores(e_va, mu, sigma)
        a_vn2 = get_anomaly_scores(e_vn2, mu, sigma)
        
        # Find the best anomaly threshold.
        tau = get_anomaly_threshold(
            a=tf.concat([a_va, a_vn2], axis=0),
            y=tf.concat([y_va, y_vn2], axis=0),
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
        
        # Make sure that the length of the time series is a multiple of the sequence length.
        x = tf.cast(x[:self.L * (len(x) // self.L)], tf.float32)
        
        # Split the time series into sequences
        xs = time_series_to_sequences(x, L=self.L)
        
        # Scale the sequences.
        xs = (xs - self.x_min) / (self.x_max - self.x_min)
        
        # Generate the reconstructions.
        rs = self.model(xs, training=False)
        
        # Transform the reconstructions back to the original scale.
        rs = self.x_min + (self.x_max - self.x_min) * rs
        
        # Transform the reconstructions back to time series.
        r = sequences_to_time_series(rs)
        
        # Calculate the reconstruction errors.
        e = tf.math.abs(x - r)

        # Calculate the anomaly scores.
        a = get_anomaly_scores(e, self.mu, self.sigma)
        
        # Derive the anomaly labels.
        y = get_anomaly_labels(a, self.tau).numpy()
        
        return r, a, y


def time_series_to_sequences(x, L):
    # Split the time series into sequences.
    return tf.concat([tf.expand_dims(x[i - L: i], axis=0) for i in range(L, L * (len(x) // L) + L, L)], axis=0)


def sequences_to_time_series(x):
    # Transform the sequences back to time series.
    return tf.concat([x[i] for i in range(len(x))], axis=0)


def reverse_sequences(x):
    # Reverse the order of the sequences.
    return tf.reverse(x, axis=[1])


def get_anomaly_scores(x, mu, sigma):
    # Calculate the anomaly scores.
    return tf.map_fn(elems=x, fn=lambda x: tf.squeeze(tf.matmul(tf.matmul(x - mu, tf.linalg.inv(sigma)), x - mu, transpose_b=True)))


def get_anomaly_labels(a, tau):
    # Derive the anomaly labels.
    return tf.where(a > tau, 1., 0.)


def get_anomaly_threshold(a, y, beta, num):
    
    # Create a tensor for storing the F-beta scores of the thresholds.
    scores = tf.TensorArray(
        element_shape=(),
        size=num,
        dynamic_size=False,
        dtype=tf.float32,
        clear_after_read=False
    )
    
    # Define the list of thresholds.
    thresholds = tf.linspace(start=float(0), stop=tf.reduce_max(a), num=num)
    
    # Loop across the thresholds.
    for i in tf.range(start=0, limit=num, delta=1):
        
        # Derive the predicted anomaly labels.
        yhat = get_anomaly_labels(a, tau=thresholds[i]).numpy()
        
        # Calculate and save the F-beta score.
        scores = scores.write(index=i, value=fbeta_score(y_true=y, y_pred=yhat, beta=beta))
    
    scores = scores.stack()
    
    # Extract the best score.
    best_score = scores[tf.argmax(scores)]
    print(f'Best score: {format(best_score.numpy(), ".6f")}')
    
    # Extract and return the best threshold.
    best_threshold = thresholds[tf.argmax(scores)]
    return best_threshold
