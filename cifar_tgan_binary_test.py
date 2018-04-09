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
from backends import get_generator, get_discriminator_binary
from data_preprocessing import get_test_data, get_train_data, extract_features
from settings import settings_binary
from triplet_utils import load_model
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
trainx, trainy, txs, tys = get_train_data(args.data_dir, args.count, args.seed_data)
nr_batches_train = int(trainx.shape[0]/args.batch_size)


testx, testy = get_test_data(args.data_dir)


# specify generative model
gen_layers = get_generator(args.batch_size, theano_rng)
gen_dat = ll.get_output(gen_layers[-1])


# specify discriminative model
disc_layers = get_discriminator_binary()


load_model(gen_layers,args.generator_pretrained)
load_model(disc_layers,args.discriminator_pretrained)

x_temp = T.tensor4()

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

print('Extracting features from test data')
test_features = extract_features(disc_layers, testx, batch_size_test)
print('Extracting features from train data')
train_features = extract_features(disc_layers, trainx, args.batch_size)

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
print("Final results Euclidian distance: ")
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
print("Final results Hamming distance: ")
print("mAP value: %.4f "% mAP)