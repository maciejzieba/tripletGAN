import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import cifar10_data
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=500)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/maciej/Pobrane/cifar-10-batches-py')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx_unl = trainx.copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)


# specify generative model
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=16, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

x_temp = T.tensor4()

temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_temp, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

init_param = th.function(inputs=[x_temp], outputs=None, updates=init_updates)

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

def getTriplets(prediction,size):
    a = prediction[0:size] # query case (positive)
    b = prediction[size:2*size] # positive case
    c = prediction[2*size:3*size] # negative

    return a,b,c

a_lab,b_lab,c_lab = getTriplets(output_before_softmax_lab,args.batch_size)

def getLossFuction(a,b,c):
    n_plus = T.sqrt(T.sum((a - b)**2, axis=1))
    n_minus = T.sqrt(T.sum((a - c)**2, axis=1))
    z = T.concatenate([n_minus.dimshuffle(0,'x'),n_plus.dimshuffle(0,'x')],axis=1)
    z = nn.log_sum_exp(z,axis=1)
    return n_plus,n_minus,z

n_plus_lab,n_minus_lab,z_lab = getLossFuction(a_lab,b_lab,c_lab)

loss_lab = -T.mean(n_minus_lab) + T.mean(z_lab)


l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]

l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)

loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)] # data based initialization
train_batch_disc = th.function(inputs=[x_lab,x_unl,lr], outputs=[loss_lab, loss_unl], updates=disc_param_updates+disc_avg_updates)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(T.square(m1-m2)) # feature matching loss
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=loss_gen, updates=gen_param_updates)

# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])

txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

def get_sim(fTrain,ftemp):

    ftemp = np.tile(ftemp, (fTrain.shape[0],1))
    dist = np.sqrt(np.sum((fTrain - ftemp)*(fTrain - ftemp),axis=1))
    ind = np.argsort(dist)

    return ind

x_temp = T.tensor4()
features = ll.get_output(disc_layers[-1], x_temp , deterministic=True)
generate_features = th.function(inputs=[x_temp ], outputs=features)
# //////////// perform training //////////////
for epoch in range(200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    nr_batches_lab = int(txs.shape[0]/args.batch_size)
    train_fx = []
    for f_id in range(nr_batches_lab):
        train_fx.append(generate_features(txs[f_id*args.batch_size:(f_id+1)*args.batch_size]))
    train_fx = np.concatenate(train_fx, axis=0)
    #train_fx = generate_features(txs)
    ind_global = []
    for k in range(np.shape(train_fx)[0]):
        ftemp = train_fx[k,:]
        ind = get_sim(train_fx,ftemp)
        ind_global.append(ind)
    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    Pos = []
    Neg = []
    for t in range(trainx_unl.shape[0]/(txs.shape[0])):
        possible_classes = np.unique(tys)
        txs_pos = np.zeros(np.shape(txs)).astype(th.config.floatX)
        txs_neg = np.zeros(np.shape(txs)).astype(th.config.floatX)
        for j in range(0,txs.shape[0]):
            label = tys[j]
            ind_pos = ind_global[j][tys[ind_global[j]]==label]
            ind_neg = ind_global[j][tys[ind_global[j]]!=label]
            txs_pos[j,:] = txs[ind_pos[-1+(-1)*t],:]
            txs_neg[j,:] = txs[ind_neg[t],:]
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
        Pos.append(txs_pos[inds])
        Neg.append(txs_neg[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    Pos = np.concatenate(Pos, axis=0)
    Neg = np.concatenate(Neg, axis=0)
    inds = rng.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy= trainy[inds]
    Pos = Pos[inds]
    Neg = Neg[inds]
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    
    if epoch == 0:
        init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    l_gen = 0.
    for t in range(nr_batches_train):
        print(t)
        temp = trainx[t*args.batch_size:(t+1)*args.batch_size]
        temp = np.concatenate((temp, Pos[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        temp = np.concatenate((temp, Neg[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        llos, lu = train_batch_disc(temp,trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += llos
        loss_unl += lu
        lg = train_batch_gen(trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        l_gen += lg

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    l_gen /= nr_batches_train

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, l_gen = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, l_gen))
    sys.stdout.flush()

np.savez('disc_params_triplet.npz', *lasagne.layers.get_all_param_values(disc_layers))
np.savez('gen_params_triplet.npz', *lasagne.layers.get_all_param_values(gen_layers))

# final testing
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
testx, testy = cifar10_data.load(args.data_dir, subset='test')

x = T.tensor4()
# extracting features
features = ll.get_output(disc_layers[-1], x, deterministic=True)
generate_features= th.function(inputs=[x], outputs=features)

batch_size_test = 100
nr_batches_test = int(testx.shape[0]/batch_size_test)
for t in range(nr_batches_test):
    if(t==0):
        test_features = generate_features(testx[t*batch_size_test:(t+1)*batch_size_test])
    else:
        test_features =  np.concatenate((test_features,generate_features(testx[t*batch_size_test:(t+1)*batch_size_test])),axis=0)

for t in range(nr_batches_train):
    if(t==0):
        train_features = generate_features(trainx[t*args.batch_size:(t+1)*args.batch_size])
    else:
        train_features =np.concatenate((train_features,generate_features(trainx[t*args.batch_size:(t+1)*args.batch_size])),axis=0)

# calculating distances
Y = cdist(test_features,train_features)
ind = np.argsort(Y,axis=1)
prec = 0.0
acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
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