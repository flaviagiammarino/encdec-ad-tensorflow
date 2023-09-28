import tensorflow as tf

class EncoderDecoder(tf.keras.models.Model):
    
    def __init__(self, L, m, c, d):
        super().__init__()

        self.encoder = tf.keras.layers.LSTM(units=c, dropout=d, return_state=True)
        self.decoder = tf.keras.layers.LSTMCell(units=c, dropout=d)
        self.outputs = tf.keras.layers.Dense(units=m)

        self.L = L
        self.m = m
        self.c = c
    
    def call(self, inputs, training=True):
        
        self.decoder.reset_dropout_mask()
        self.decoder.reset_recurrent_dropout_mask()
        
        _, he, ce = self.encoder(inputs)
    
        hd = tf.identity(he)
        cd = tf.identity(ce)
    
        r = tf.TensorArray(
            element_shape=(inputs.shape[0], self.m),
            size=self.L,
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )
    
        r = r.write(
            index=self.L - 1,
            value=self.outputs(hd)
        )
    
        for t in tf.range(start=self.L - 2, limit=-1, delta=-1):
                
            _, [hd, cd] = self.decoder(
                inputs=inputs[:, t + 1, :] if training else r.read(index=t + 1),
                states=[hd, cd],
                training=training
            )
        
            r = r.write(
                index=t,
                value=self.outputs(hd)
            )
        
        return tf.transpose(r.stack(), (1, 0, 2))