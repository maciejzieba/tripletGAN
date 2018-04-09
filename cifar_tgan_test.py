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
import plotting
import cifar10_data
from settings import settings_binary
from validation_utils import hamming_dist, mean_average_precison
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

# settings
args = settings_binary()

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))


# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
nr_batches_train = int(trainx.shape[0]/args.batch_size)

testx, testy = cifar10_data.load(args.data_dir, subset='test')
inds = rng_data.permutation(testx.shape[0])
testx = testx[inds]
testy = testy[inds]
testx = testx[0:100]
testy = testy[0:100]

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

path_gen = 'gen_params_triplet.npz'
path_disc = 'disc_params_triplet.npz'

x_temp = T.tensor4()

with np.load(path_disc) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(disc_layers, param_values)

with np.load(path_gen) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(gen_layers, param_values)


# Test generator in sampling procedure
samplefun = th.function(inputs=[],outputs=gen_dat)
sample_x = samplefun()
img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title='CIFAR10 samples')
plotting.plt.savefig("cifar_tgan_sample.png")

features = ll.get_output(disc_layers[-1], x_temp , deterministic=True)
generateTestF = th.function(inputs=[x_temp ], outputs=features)


batch_size_test = 100
nr_batches_test = int(testx.shape[0]/batch_size_test)

print('Extracting features from test data')

for t in range(nr_batches_test):
    if t == 0:
        test_features = generateTestF(testx[t*batch_size_test:(t+1)*batch_size_test])
    else:
        test_features =  np.concatenate((test_features,generateTestF(testx[t*batch_size_test:(t+1)*batch_size_test])),axis=0)

print('Extracting features from train data')

for t in range(nr_batches_train):
    if(t==0):
        train_features = generateTestF(trainx[t*args.batch_size:(t+1)*args.batch_size])
    else:
        train_features =np.concatenate((train_features,generateTestF(trainx[t*args.batch_size:(t+1)*args.batch_size])),axis=0)


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


# calculating distances
test_features[test_features >=0.5] = 1
test_features[test_features < 0.5] = 0


train_features[train_features >=0.5] = 1
train_features[train_features < 0.5] = 0


Y = hamming_dist(test_features,train_features)
mAP = mean_average_precison(testy,trainy,Y)
print(mAP)