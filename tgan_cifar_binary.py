import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
import nn
import sys

# settings
from backends import get_generator, get_discriminator_binary
from data_preprocessing import get_train_data
from settings import settings_binary
from triplet_utils import load_model, modify_indexes, get_sim

args = settings_binary()
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))


# load CIFAR-10
trainx, trainy, txs, tys = get_train_data(args.data_dir, args.count, args.seed_data)
trainx_unl = trainx.copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)


# specify generative model
gen_layers = get_generator(args.batch_size, theano_rng)
gen_dat = ll.get_output(gen_layers[-1])


# specify discriminative model
disc_layers = get_discriminator_binary()
disc_params = ll.get_all_params(disc_layers, trainable=True)


# you can use pretrained models
if args.use_pretrained:
    load_model(gen_layers,args.generator_pretrained)
    load_model(disc_layers,args.discriminator_pretrained)


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
output_before_softmax_unl = ll.get_output(disc_layers[-2], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)


def get_triplets(prediction,size):
    a = prediction[0:size] # query case (positive)
    b = prediction[size:2*size] # positive case
    c = prediction[2*size:3*size] # negative

    return a,b,c


a_lab,b_lab,c_lab = get_triplets(output_before_softmax_lab,args.batch_size)


def get_loss(a,b,c):
    n_plus = T.sqrt(0.01+T.sum((a - b)**2, axis=1))
    n_minus = T.sqrt(0.01+T.sum((a - c)**2, axis=1))
    z = T.concatenate([n_minus.dimshuffle(0,'x'),n_plus.dimshuffle(0,'x')],axis=1)
    z = nn.log_sum_exp(z,axis=1)
    return n_plus,n_minus,z


n_plus_lab,n_minus_lab,z_lab = get_loss(a_lab,b_lab,c_lab)


loss_lab = -T.mean(n_minus_lab) + T.mean(z_lab)


l_unl = output_before_softmax_unl[:,0]
l_gen = output_before_softmax_gen[:,0]


loss_unl = -0.5*T.mean(T.log(0.9998*T.nnet.sigmoid(l_unl) + 0.0001)) - \
           0.5*T.mean(T.log(1-0.9998*T.nnet.sigmoid(l_gen)))


# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params,loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)] # data based initialization
train_batch_disc = th.function(inputs=[x_lab,x_unl,lr], outputs=[loss_lab, loss_unl], updates=disc_param_updates)


# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-3], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-3], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(T.square(m1-m2)) # feature matching loss
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=loss_gen, updates=gen_param_updates)

x_temp = T.tensor4()
features = ll.get_output(disc_layers[-1], x_temp , deterministic=True)
generate_features = th.function(inputs=[x_temp], outputs=features)
# //////////// perform training //////////////
for epoch in range(1):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    nr_batches_lab = int(txs.shape[0]/args.batch_size)
    train_fx = []
    for f_id in range(nr_batches_lab):
        train_fx.append(generate_features(txs[f_id*args.batch_size:(f_id+1)*args.batch_size]))
    train_fx = np.concatenate(train_fx, axis=0)
    ind_global = []
    for k in range(np.shape(train_fx)[0]):
        ftemp = train_fx[k,:]
        ind = get_sim(train_fx,ftemp)
        ind_global.append(ind)
    trainx = []
    trainy = []
    Pos = []
    Neg = []
    for t in range(int(trainx_unl.shape[0]/(txs.shape[0]))):
        possible_classes = np.unique(tys)
        txs_pos = np.zeros(np.shape(txs)).astype(th.config.floatX)
        txs_neg = np.zeros(np.shape(txs)).astype(th.config.floatX)
        for j in range(txs.shape[0]):
            label = tys[j]
            ind_pos = ind_global[j][tys[ind_global[j]]==label]
            ind_neg = ind_global[j][tys[ind_global[j]]!=label]
            ind_pos = modify_indexes(ind_pos, args.triplet_positives_ordering)
            ind_neg = modify_indexes(ind_neg, args.triplet_negatives_ordering)
            txs_pos[j,:] = txs[ind_pos[-1+(-1)*t],:]
            txs_neg[j,:] = txs[ind_neg[-1+(-1)*t],:]
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
    
    if epoch == 0 and not args.use_pretrained:
        print(trainx.shape)
        init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    l_gen = 0.
    for t in range(nr_batches_train):
        temp = trainx[t*args.batch_size:(t+1)*args.batch_size]
        temp = np.concatenate((temp, Pos[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        temp = np.concatenate((temp, Neg[t*args.batch_size:(t+1)*args.batch_size]),axis=0)
        llos, lu = train_batch_disc(temp,trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += llos
        loss_unl += lu
        print(t)
        lg = train_batch_gen(trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        l_gen += lg

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    l_gen /= nr_batches_train
    # test
    test_err = 0.
    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, l_gen = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, l_gen))
    sys.stdout.flush()


    if epoch == 0:
        results = np.array([epoch, time.time()-begin, loss_lab, loss_unl, l_gen]).reshape((1,5))
    else:
        results = np.concatenate((results,np.array([epoch, time.time()-begin, loss_lab, loss_unl, l_gen]).reshape((1,5))),axis=0)

    np.savetxt("result.txt",results,fmt='%.4f',delimiter=',')

    np.savez(args.discriminator_out, *lasagne.layers.get_all_param_values(disc_layers))
    np.savez(args.generator_out, *lasagne.layers.get_all_param_values(gen_layers))
