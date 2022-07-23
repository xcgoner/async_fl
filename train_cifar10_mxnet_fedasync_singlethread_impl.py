import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

from os import listdir
import os.path
import argparse

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
parser.add_argument("--valdir", type=str, help="dir of the val data", required=True)
parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=50)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=100)
parser.add_argument("-v", "--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("-n", "--nsplit", type=int, help="number of split", default=40)
parser.add_argument("-l", "--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--momentum", type=float, help="momentum", default=0)
# instead of define the weight of the regularization, we define lr * regularization weight, this option is equivalent to \gamma \times \rho in the paper
parser.add_argument("--rho", type=float, help="regularization \times lr", default=0.00001)
parser.add_argument("--alpha", type=float, help="mixing hyperparameter", default=0.9)
parser.add_argument("-o", "--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=20)
parser.add_argument("-i", "--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("--alpha-decay", type=float, help="alpha decay rate", default=0.5)
parser.add_argument("--alpha-decay-epoch", type=str, help="alpha decay epoch", default='400')
parser.add_argument("--alpha-adaptive", type=float, help="adaptive mixing hyperparameter", default=0)
parser.add_argument("--alpha-adaptive2", type=float, help="adaptive mixing hyperparameter2", default=1)
parser.add_argument("--alpha-type", type=str, help="type of adaptive alpha", default='none')
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--seed", type=int, help="random seed", default=733)
parser.add_argument("--max-delay", type=int, help="maximum of global delay", default=10)
 
args = parser.parse_args()

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# set random seed
mx.random.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# set path
data_dir = os.path.join(args.dir, 'dataset_split_{}'.format(args.nsplit))
train_dir = os.path.join(data_dir, 'train')
# path to validation data
val_dir = os.path.join(args.valdir, 'val')

training_files = []
for filename in sorted(listdir(train_dir)):
    absolute_filename = os.path.join(train_dir, filename)
    training_files.append(absolute_filename)

context = mx.cpu()

classes = args.classes

# load training data
def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

# load validation data
def get_val_train_batch(data_dir):
    test_filename = os.path.join(data_dir, 'train_data.pkl')
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

def get_val_val_batch(data_dir):
    test_filename = os.path.join(data_dir, 'val_data.pkl')
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

model_name = args.model

if model_name == 'default':
    net = gluon.nn.Sequential()
    with net.name_scope():
        #  First convolutional layer
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        #  Second convolutional layer
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Third convolutional layer
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten and apply fullly connected layers
        net.add(gluon.nn.Flatten())
        # net.add(gluon.nn.Dense(512, activation="relu"))
        # net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(rate=0.25))
        net.add(gluon.nn.Dense(classes))
else:
    model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
    net = get_model(model_name, **model_kwargs)

# initialization
if model_name.startswith('cifar') or model_name == 'default':
    net.initialize(mx.init.Xavier(), ctx=context)
else:
    net.initialize(mx.init.MSRAPrelu(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0

# SGD optimizer
optimizer = 'sgd'
lr = args.lr
optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0.0001}
# optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

# lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
alpha_decay_epoch = [int(i) for i in args.alpha_decay_epoch.split(',')]

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_cross_entropy = mx.metric.CrossEntropy()

# warmup
print('warm up', flush=True)
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
trainer.set_learning_rate(0.01)
[train_X, train_Y] = get_train_batch(training_files[0])
train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
for local_epoch in range(5):
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)

nd.waitall()

params_prev = [param.data().copy() for param in net.collect_params().values()]
params_prev_list = [params_prev]
server_ts = 0
ts_list = [server_ts]

nd.waitall()


rho = args.rho
alpha = args.alpha

train_data_list = []
for i in range(args.nsplit):
    [train_X, train_Y] = get_train_batch(training_files[i])
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
    train_data_list.append(train_data)

[val_train_X, val_train_Y] = get_val_train_batch(val_dir)
val_train_dataset = mx.gluon.data.dataset.ArrayDataset(val_train_X, val_train_Y)
val_train_data = gluon.data.DataLoader(val_train_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

[val_val_X, val_val_Y] = get_val_val_batch(val_dir)
val_val_dataset = mx.gluon.data.dataset.ArrayDataset(val_val_X, val_val_Y)
val_val_data = gluon.data.DataLoader(val_val_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

sum_delay = 0
tic = time.time()
for epoch in range(args.epochs):

    # alpha decay
    if epoch in alpha_decay_epoch:
        alpha = alpha * args.alpha_decay

    # obtain previous model
    model_idx = random.randint(0, len(params_prev_list)-1)
    params_prev = params_prev_list[model_idx]
    for param, param_prev in zip(net.collect_params().values(), params_prev):
        if param.grad_req != 'null':
            weight = param.data()
            weight[:] = param_prev

    params_prev = [param.data().copy() for param in net.collect_params().values()]
    if rho > 0:
        # param_prev is the initial model, pulled at the beginning of the epoch
        # param_prev is rescaled by rho
        for param, param_prev in zip(net.collect_params().values(), params_prev):
            if param.grad_req != 'null':
                param_prev[:] = param_prev * rho
    nd.waitall()

    worker_ts = ts_list[model_idx]

    # train on worker
    # randomly select a local dataset/worker
    train_data = random.choice(train_data_list)
    # reset optimizer
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    trainer.set_learning_rate(lr)     

    # local epoch
    for local_epoch in range(args.iterations):
        for i, (data, label) in enumerate(train_data):
            with ag.record():
                outputs = net(data)
                loss = loss_func(outputs, label)
            loss.backward()
            # add regularization
            if rho > 0:
                for param, param_prev in zip(net.collect_params().values(), params_prev):
                    if param.grad_req != 'null':
                        weight = param.data()
                        # param_prev is already rescaled by rho before
                        weight[:] = weight * (1-rho) + param_prev
            trainer.step(args.batchsize)
    
    nd.waitall()

    # update on server 
    worker_delay = epoch - worker_ts
    sum_delay += worker_delay
    if args.alpha_type == 'power':
        if args.alpha_adaptive > 0:
            alpha_factor = 1 / math.pow(worker_delay + 1.0, args.alpha_adaptive)
        else:
            alpha_factor = 1
    elif args.alpha_type == 'exp':
        alpha_factor = math.exp(-worker_delay * args.alpha_adaptive)
    elif args.alpha_type == 'sigmoid':
        # a soft-gated function in the range of (1/a, 1]
        # maximum value
        a = args.alpha_adaptive2
        # slope
        c = args.alpha_adaptive
        b = math.exp(- c * worker_delay)
        alpha_factor = (1 - b) / a + b
    elif args.alpha_type == 'hinge':
        if worker_delay <= args.alpha_adaptive2:
            alpha_factor = 1
        else:
            alpha_factor = 1.0 / ( (worker_delay - args.alpha_adaptive2) * args.alpha_adaptive + 1)
            # alpha_factor = math.exp(- (worker_delay-args.alpha_adaptive2) * args.alpha_adaptive)
    else:
        alpha_factor = 1
    alpha_scaled = alpha * alpha_factor
    for param, param_server in zip(net.collect_params().values(), params_prev_list[-1]):
        weight = param.data()
        weight[:] = param_server * (1-alpha_scaled) + weight * alpha_scaled
    # push
    params_prev_list.append([param.data().copy() for param in net.collect_params().values()])
    ts_list.append(epoch+1)
    # pop
    if len(params_prev_list) > args.max_delay:
        del params_prev_list[0]
        del ts_list[0]
    nd.waitall()
    
    # validation
    if  epoch % args.interval == 0 or epoch == args.epochs-1:
        acc_top1.reset()
        acc_top5.reset()
        train_cross_entropy.reset()
        # get accuracy on testing data
        for i, (data, label) in enumerate(val_val_data):
            outputs = net(data)
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        # get cross entropy loss on traininig data
        for i, (data, label) in enumerate(val_train_data):
            outputs = net(data)
            train_cross_entropy.update(label, nd.softmax(outputs))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        _, crossentropy = train_cross_entropy.get()

        logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, lr=%f, rho=%f, alpha=%f, max_delay=%d, mean_delay=%f, time=%f' % (epoch, top1, top5, crossentropy, trainer.learning_rate, rho, alpha, args.max_delay, sum_delay*1.0/(epoch+1), time.time()-tic))
        tic = time.time()
        
        nd.waitall()



            






