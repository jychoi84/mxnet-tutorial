from __future__ import print_function
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet import gluon

ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1, in_units=2))

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

epochs = 1
smoothing_constant = .01
moving_loss = 0 
niter = 0 
loss_seq = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label  = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)

        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant * curr_loss)

        #est_loss = moving_loss / (1 - (1 - smoothing_constant)**niter)
        est_loss = moving_loss
        loss_seq.append(est_loss)

    print("Epoch %s. Moving avg of MSE: %s" % (e, est_loss))

import matplotlib 
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(8, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')
plt.semilogy(range(niter), loss_seq, '.')

plt.grid(True, which="both")
plt.xlabel('iteration', fontsize=14)
plt.ylabel('est loss', fontsize=14)
#plt.show()

params = net.collect_params()
print('The type of "params" is a ', type(params))

for param in params.values():
    print(param.name, param.data())
