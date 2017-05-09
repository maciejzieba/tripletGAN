import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--count', type=int, default=10)
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# specify generative model
noise = theano_rng.uniform(size=(args.batch_size, 100))
gen_layers = [LL.InputLayer(shape=(args.batch_size, 100), input_var=noise)]
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.l2normalize(LL.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = LL.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
layers = [LL.InputLayer(shape=(None, 28**2))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.DenseLayer(layers[-1], num_units=1000))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=500))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=16, nonlinearity=None, train_scale=True))

x_lab = T.matrix()
x_unl = T.matrix()

temp = LL.get_output(gen_layers[-1], init=True)
temp = LL.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_lab = LL.get_output(layers[-1], x_lab, deterministic=False)
output_unl = LL.get_output(layers[-1], x_unl, deterministic=False)
output_fake = LL.get_output(layers[-1], gen_dat, deterministic=False)

def get_triplets(prediction,size):
    a = prediction[0:size] # query case (positive)
    b = prediction[size:2*size] # positive case
    c = prediction[2*size:3*size] # negative

    return a,b,c

a_lab,b_lab,c_lab = get_triplets(output_lab,args.batch_size)


def loss_labeled(a,b,c):
    n_plus = T.sqrt(T.sum((a - b)**2, axis=1));
    n_minus = T.sqrt(T.sum((a - c)**2, axis=1));
    z = T.concatenate([n_minus.dimshuffle(0,'x'),n_plus.dimshuffle(0,'x')],axis=1)
    z = nn.log_sum_exp(z,axis=1)
    return n_plus,n_minus,z

n_plus_lab,n_minus_lab,z_lab = loss_labeled(a_lab,b_lab,c_lab)

# defning triplet loss function
loss_lab = -T.mean(n_minus_lab) + T.mean(z_lab)

# defining unlabelled loss
loss_unl = -0.5*T.mean(nn.log_sum_exp(output_unl)) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_unl))) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_fake)))

# defining feature matching loss for generator training
mom_gen = LL.get_output(layers[-1], gen_dat)
mom_real = LL.get_output(layers[-1], x_unl)
loss_gen = T.mean(T.square(T.mean(mom_gen, axis=0) -T.mean( mom_real, axis=0)))

# Theano functions for training and testing
lr = T.scalar()
disc_params = LL.get_all_params(layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = LL.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)

train_batch_disc = th.function(inputs=[x_lab,x_unl,lr], outputs=[loss_lab, loss_unl], updates=disc_param_updates+disc_avg_updates)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=loss_gen, updates=gen_param_updates)

# load MNIST data
data = np.load('mnist.npz')
samplefun = th.function(inputs=[],outputs=gen_dat)
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)

# select labeled data
inds = data_rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

init_param(trainx[:500]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.0003

def get_sim(fTrain,ftemp):

    ftemp = np.tile(ftemp, (fTrain.shape[0],1))
    dist = np.sqrt(np.sum((fTrain - ftemp)*(fTrain - ftemp),axis=1))
    ind = np.argsort(dist)

    return ind

x_temp = T.matrix()
features = LL.get_output(layers[-1], x_temp , deterministic=True)
generateTestF = th.function(inputs=[x_temp ], outputs=features)
for epoch in range(10):
    begin = time.time()

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    Pos = []
    Neg = []
    possible_classes = np.unique(tys);
    for i in range(len(possible_classes)):
        class_lab = possible_classes[i]
        q_temp = txs[tys==class_lab]
        p_temp = txs[tys==class_lab]
        n_temp = txs[tys!=class_lab]
        for tp in range(args.count):
            for tn in range(txs.shape[0]-args.count):
                trainx.append(q_temp)
                trainy.append(tys[tys==class_lab])
                Pos.append(np.roll(p_temp, tp, axis=0))
                Neg.append(np.roll(n_temp, tn, axis=0))
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    Pos = np.concatenate(Pos, axis=0)
    Neg = np.concatenate(Neg, axis=0)
    inds = rng.permutation(trainx.shape[0])

    inds = inds[0:trainx_unl.shape[0]]
    trainx = trainx[inds]
    trainy= trainy[inds]
    Pos = Pos[inds]
    Neg = Neg[inds]
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    # train
    loss_lab = 0.
    loss_unl = 0.
    loss_g = 0.
    for t in range(nr_batches_train):
        temp = trainx[t*args.batch_size:(t+1)*args.batch_size]
        temp = np.concatenate((temp, Pos[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        temp = np.concatenate((temp, Neg[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        ll, lu = train_batch_disc(temp,trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += ll
        loss_unl += lu

        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_g += e
    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    loss_g /= nr_batches_train

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, loss_gen = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl,loss_g ))
    sys.stdout.flush()
    np.savez('disc_params_MNIST.npz', *[p.get_value() for p in disc_params])
    np.savez('gen_params_MINST.npz', *[p.get_value() for p in gen_params])

# final testing
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)

x = T.matrix()
# extracting features
features = LL.get_output(layers[-1], x, deterministic=True)
generate_features= th.function(inputs=[x], outputs=features)
test_features = generate_features(testx)

for t in range(nr_batches_train):
    if(t==0):
        train_features = generate_features(trainx[t*args.batch_size:(t+1)*args.batch_size])
    else:
        train_features =np.concatenate((train_features,generate_features(trainx[t*args.batch_size:(t+1)*args.batch_size])),axis=0)

# calculating distances
Y = cdist(test_features,train_features)
ind = np.argsort(Y,axis=1)
prec = 0.0;
acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
# calculating statistics
for k in range(np.shape(test_features)[0]):
    class_values = trainy[ind[k,:]]
    y_true = (testy[k] == class_values)
    y_scores = np.arange(y_true.shape[0],0,-1)
    ap = average_precision_score(y_true, y_scores)
    prec = prec + ap
    for n in range(len(acc)):
        a = class_values[0:(n+1)]
        counts = np.bincount(a)
        b = np.where(counts==np.max(counts))[0]
        if testy[k] in b:
            acc[n] = acc[n] + (1.0/float(len(b)))
prec = prec/float(np.shape(test_features)[0])
acc= [x / float(np.shape(test_features)[0]) for x in acc]
print("Final results: ")
print("mAP value: %.4f "% prec)
for k in range(len(acc)):
    print("Accuracy for %d - NN: %.2f %%" % (k+1,100*acc[k]) )