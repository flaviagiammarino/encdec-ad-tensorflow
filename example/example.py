import numpy as np

from encdec_ad_tensorflow.model import EncDecAD
from encdec_ad_tensorflow.plots import plot

m = 2       # Number of time series
N = 1000    # Length of each time series

# Generate the normal time series
xn = np.zeros((N, 2 * m))
t = np.linspace(0, 1, N)
c = np.cos(2 * np.pi * (10 * t - 0.5))
s = np.sin(2 * np.pi * (20 * t - 0.5))
for i in range(2 * m):
    a = np.random.uniform(10, 20)
    b = np.random.uniform(10, 20)
    xn[:, i] = a * c + b * s + np.random.normal(size=N)

# Generate the anomalous time series
xa = xn.copy()
for j in range(N):
    if np.random.uniform() > 0.8:
        xa[j, :] = np.random.randint(low=30, high=70)

# Split the time series
xn_train, xn_test = xn[:, :m], xn[:, m:]
xa_train, xa_test = xa[:, :m], xa[:, m:]

# Fit the model
model = EncDecAD(
    m=2,          # Number of time series.
    L=100,        # Number of time steps (window size or sequence length).
    c=128,        # Number of hidden units.
    beta=0.1      # F1-score's beta.
)

model.fit(
    xn_train,
    xa_train,
    learning_rate=0.001,
    batch_size=32,
    max_epochs=100,
    early_stopping_start_epoch=50,
    early_stopping_patience=10,
    verbose=1
)

# Generate the training set predictions.
rn_train, an_train, yn_train = model.predict(xn_train)
ra_train, aa_train, ya_train = model.predict(xa_train)

# Generate the test set predictions.
rn_test, an_test, yn_test = model.predict(xn_test)
ra_test, aa_test, ya_test = model.predict(xa_test)

# Evaluate the results.
print(f'Training Accuracy: {((yn_train == 0).sum() + (ya_train == 1).sum()) / (len(yn_train) + len(ya_train))}')
print(f'Test Accuracy: {((yn_test == 0).sum() + (ya_test == 1).sum()) / (len(yn_test) + len(ya_test))}')

# Plot the results.
fig = plot(xn=xn_test, xa=xa_test, rn=rn_test, ra=ra_test, an=an_test, aa=aa_test, tau=model.tau.numpy())
fig.write_image('results.png', scale=4, height=900, width=400 * m)
