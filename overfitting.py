from __future__ import print_function
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
ctx = mx.cpu()

import matplotlib
import matplotlib.pyplot as plt

mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(mnist['train_data'][:num_examples], 
                                                           mnist['train_label'][:num_examples].astype(np.float32)), 
                                                            batch_size, shuffle = True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(mnist['test_data'][:num_examples], 
                                                          mnist['test_label'][:num_examples].astype(np.float32)),
                                                            batch_size, shuffle=False)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx = ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01, 'wd':0.001})

def evaluate_accuracy(data_iterator, net, loss_fun):
    acc = mx.metric.Accuracy()
    loss_avg = 0 
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        loss = loss_fun(output, label)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds = predictions, labels = label)
        loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar()/(i+1)
    return acc.get()[1], loss_avg

def plot_learningcurves(loss_tr, loss_ts, acc_tr, acc_ts):
    xs = list(range(len(loss_tr)))

    f = plt.figure(figsize= (12, 6))
    fg1 = f.add_subplot(121)
    fg2 = f.add_subplot(122)

    fg1.set_xlabel('epoch',fontsize=14)
    fg1.set_title('Comparing loss functions')
    fg1.semilogy(xs, loss_tr)
    fg1.semilogy(xs, loss_ts)
    fg1.grid(True,which="both")

    fg1.legend(['training loss', 'testing loss'],fontsize=14)

    fg2.set_title('Comparing accuracy')
    fg1.set_xlabel('epoch',fontsize=14)
    fg2.plot(xs, acc_tr)
    fg2.plot(xs, acc_ts)
    fg2.grid(True,which="both")
    fg2.legend(['training accuracy', 'testing accuracy'],fontsize=14)
    
    plt.show()


epochs = 400
moving_loss = 0.
niter = 0 

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])

        niter += 1 
        moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
        est_loss = moving_loss / (1 - 0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net, loss)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net, loss)

    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 20 == 0 :
        print("Completed epoch %s. Train Loss: %s, Test Loss: %s, Train_acc: %s, Test_acc: %s" 
              % (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

plot_learningcurves(loss_seq_train, loss_seq_test, acc_seq_train, acc_seq_test)