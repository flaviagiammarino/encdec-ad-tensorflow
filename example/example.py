import numpy as np

from encdec_ad_tensorflow.model import EncDecAD
from encdec_ad_tensorflow.plots import plot

# Generate some time series
N = 2000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=N)
a = 150 + 15 * np.cos(2 * np.pi * (15 * t - 0.5)) + 0.15 * e[:, 0]
b = 200 + 20 * np.cos(2 * np.pi * (20 * t - 0.5)) + 0.20 * e[:, 1]
x = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

# Fit the model
model = EncDecAD(
    x=x,
    units=100,
    timesteps=200
)

model.fit(
    learning_rate=0.001,
    batch_size=32,
    epochs=500,
    verbose=True
)

# Add some anomalies
a = np.random.choice(a=[200 * p for p in range(1, N // 200 - 1)], replace=False, size=4)
b = np.random.randint(low=15, high=25, size=4)
for i in range(4):
    x[a[i]: a[i] + b[i], :] = np.random.randint(low=60, high=70, size=2)

# Score the anomalies
x_hat, scores = model.predict(x=x)

# Plot the anomalies
fig = plot(x=x, x_hat=x_hat, scores=scores, quantile=0.97)
fig.write_image('results.png', scale=4, height=900, width=700)