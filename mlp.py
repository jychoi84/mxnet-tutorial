from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

ctx = mx.cpu()

#ctx = mx.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10 

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform = transform), batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform = transform), batch_size, shuffle=False)

num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx = ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds=predictions, labels = label)
    return acc.get()[1]

#acc = evaluate_accuracy(test_data, net)
#print(acc)

epochs = 10
moving_loss = 0.
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(batch_size)

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch: %s. loss: %s, Train acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

import sys
sys.exit(1)

import matplotlib.pyplot as plt

def model_predict(net, data):
    output = net(data)
    return nd.argmax(output, axis = 1)

sample_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform), 10, shuffle = True)

for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(ctx)
    print(data.shape)
    im = nd.transpose(data, (1, 0, 2, 3))
    im = nd.reshape(im, (28, 10*28, 1))
    imtiles = nd.tile(im, (1, 1, 3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred = model_predict(net, data.reshape((-1, 784)))
    print('model prediction are:', pred)
    break

    

