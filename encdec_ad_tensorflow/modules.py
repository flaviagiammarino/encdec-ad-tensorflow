import tensorflow as tf

class EncoderDecoder(tf.keras.layers.Layer):
    
    def __init__(self, units):
        
        '''
        LSTM Encoder-Decoder layer, see Section 2.1 of the EncDec-AD paper.

        Parameters:
        __________________________________
        units: int.
            Number of hidden units.
        '''
        
        self.units = units
        self.encoder = None
        self.decoder = None
        self.outputs = None
        super(EncoderDecoder, self).__init__()
    
    def build(self, input_shape):

        if self.encoder is None:
            self.encoder = tf.keras.layers.LSTM(units=self.units)
        
        if self.decoder is None:
            self.decoder = tf.keras.layers.LSTMCell(units=self.units)

        if self.outputs is None:
            self.outputs = tf.keras.layers.Dense(units=input_shape[-1])

    def call(self, inputs, training=True):
    
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Model inputs (actual time series), tensor with shape (samples, timesteps, features) where samples
            is the batch size, timesteps is the number of time steps and features is the number of time series.
        
        training: bool.
            Whether the call is in training mode (True) or inference mode (False).
            
        Returns:
        __________________________________
        outputs: tf.Tensor.
            Model outputs (reconstructed time series), tensor with shape (samples, timesteps, features) where samples
            is the batch size, timesteps is the number of time steps and features is the number of time series.
        '''
        
        # Initialize the outputs.
        y = tf.TensorArray(
            element_shape=(inputs.shape[0], inputs.shape[2]),
            size=inputs.shape[1],
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )
        
        # Get the inputs.
        x = tf.cast(inputs, dtype=tf.float32)
        
        # Update the encoder states.
        he = self.encoder(x)

        # Initialize the decoder states.
        hd = tf.identity(he)
        cd = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        
        # Update the decoder states.
        for t in tf.range(start=inputs.shape[1] - 1, limit=-1, delta=-1):
            y = y.write(index=t, value=self.outputs(hd))
            hd, [hd, cd] = self.decoder(states=[hd, cd], inputs=x[:, t, :] if training else y.read(index=t))

        # Return the outputs.
        return tf.transpose(y.stack(), (1, 0, 2))
