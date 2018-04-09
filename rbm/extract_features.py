from __future__ import print_function
import argparse
import cPickle
import sys
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T
import RBM
import nn
from validation_utils import average_precision_score
import cifar10_data
from scipy.spatial.distance import cdist

# ##################### Build Restricted Boltzmann Machine #######################
def build_rbm(input_var=None,d_x=21,d_y=21):

    l_in = lasagne.layers.InputLayer(shape=(None, 1, d_x, d_y),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid = RBM.RBM(l_in_drop,num_units=128,kGibbs=1)

    return l_hid


# ############################# Batch iterator ###############################
def iterate_minibatches(inputs,X_q,X_p,X_n, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt],X_q[excerpt],X_p[excerpt], X_n[excerpt]


def main(model='rbm', num_epochs=50,for_training=True,n_avg = 1,model_name = 'rbm_noise.pkl'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1)
    parser.add_argument('--seed_data', default=1)
    parser.add_argument('--count', default=500)
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--unlabeled_weight', type=float, default=1.)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--data_dir', type=str, default='/home/maciej/Pobrane/cifar-10-batches-py')
    args = parser.parse_args()

    rng_data = np.random.RandomState(args.seed_data)
    rng = np.random.RandomState(args.seed)
    X_train = np.load('train_featuresCIFAR_ordered2.npy')
    print(X_train.shape)
    '''m = np.mean(X_train);
    s = np.std(X_train)
    X_train = (X_train - m)/s'''
    '''X_train[X_train>0.5] = 1;
    X_train[X_train<0.5] = 0;'''
    X_train_ver = X_train.copy()
    X_train = X_train.reshape((X_train.shape[0], 1,1,X_train.shape[1]))
    X_train_l = X_train.copy()
    X_train_out = X_train
    X_val = X_train;
    X_test = np.load('test_featuresCIFAR_ordered2.npy');
    print(X_test.shape)
    #X_test = (X_test -m)/s
    '''X_test[X_test>0.5] = 1;
    X_test[X_test<0.5] = 0;'''
    X_test_ver = X_test.copy()
    X_test = X_test.reshape((X_test.shape[0],1,1,X_test.shape[1]))

    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    d_x =  X_train.shape[2]
    d_y = X_train.shape[3]

    trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
    inds = rng_data.permutation(trainx.shape[0])
    X_train_l = X_train_l[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(X_train_l[trainy==j][:args.count])
        tys.append(trainy[trainy==j][:args.count])

    '''for j in range(10):
        txs.append(trainx[trainy==j])
        tys.append(trainy[trainy==j])'''

    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    # Create RBM
    if model == 'rbm':
        if(for_training):
            rbm = build_rbm(input_var,d_x,d_y)
        else:
            rbm = cPickle.load(open(model_name ))
            rbm = rbm[0]
    else:
        print("Unrecognized model type %r." % model)

    # Create a loss expression for training, i.e., free energy:
    Xk = rbm.get_output_for( input_var, kGibbs=1 )
    i_v = input_var.flatten(2)
    x_p = T.tensor4()
    x_q = T.tensor4()
    x_n = T.tensor4()
    h_p_before = rbm.sample_h_given_x(x_p)[0]
    h_p= 2.0*rbm.sample_h_given_x(x_p)[0] - 1.0
    h_q= 2.0*rbm.sample_h_given_x(x_q)[0] - 1.0
    h_n= 2.0*rbm.sample_h_given_x(x_n)[0] - 1.0

    reg_lambda = 1.0
    rbm_lambda = 1.0
    rbm_lab = 1.0
    labmda_margin = 16.0
    lr = 0.0001
    mom = 0.5

    n_plus = T.sum((h_p*h_q), axis=1)
    n_minus = T.sum((h_p*h_n), axis=1)
    factor = n_plus-n_minus-labmda_margin
    #reg_term = 1.0*T.mean(T.sum((h_p-T.sgn(T.log(h_p_before)-T.log(1.0-h_p_before)))**2, axis=1))
    reg_term = reg_lambda*T.mean(T.sum((h_p-T.sgn(h_p))**2, axis=1))
    loss_lab = -1.0*T.mean((factor-T.nnet.softplus(factor))) + reg_term
    #lam = 0.000003;
    lam = 0.0;
    loss_rbm = (T.mean( rbm.free_energy(i_v) ) - T.mean( rbm.free_energy(Xk) ))
    loss = rbm_lambda*loss_rbm + rbm_lab*loss_lab+lam*(rbm.W** 2).sum() #rbm.objective( input_var, Xk )

    #Define params
    params = lasagne.layers.get_all_params(rbm, trainable=True)
    #Define updates
    '''updates = updatesPWr.nesterov_momentum(
            loss, params, learning_rate=0.1, momentum=0.099, as_constant=Xk)

    train_fn = theano.function([input_var,x_q,x_p,x_n], loss, updates=updates)'''

    updates= nn.adam_updates(params,loss, lr=lr, mom1=mom,as_constant=Xk)
    train_fn = theano.function([input_var,x_q,x_p,x_n], [loss,loss_rbm,loss_lab], updates=updates)
    #Define a function for validation and testing as a reconstruction error
    Xr = rbm.predict(i_v) #return reconstruction of inputs (with 1-step Gibbs sampling) as probabilities
    reconstrunction_error = T.mean(T.sum((Xr-i_v)*(Xr-i_v), axis=1), dtype=theano.config.floatX)
    val_fn = theano.function([i_v], reconstrunction_error)

    # Finally, launch the training loop.
    print("Starting training...")
    loss_unsup = []
    loss_sup = []
    loss_total = []
    # We iterate over epochs:
    #for_training = False;
    def get_sim(fTrain,ftemp):
        ftemp = np.tile(ftemp, (fTrain.shape[0],1))
        dist = np.sqrt(np.sum((fTrain - ftemp)*(fTrain - ftemp),axis=1))
        ind = np.argsort(dist)

        return ind
    samp = rbm.sample_h_given_x(i_v)
    samFun = theano.function([input_var], samp)

    if(for_training):
        for epoch in range(num_epochs):
            train_fx = samFun(txs)[0]
            train_fx[train_fx>0.5] = 1;
            train_fx[train_fx<0.5] = 0;
            '''ind_global = []
            for k in range(np.shape(train_fx)[0]):
                ftemp = train_fx[k,:]
                ind = get_sim(train_fx,ftemp)
                ind_global.append(ind)'''
            Y = cdist(train_fx,train_fx,'hamming')
            ind_global = np.argsort(Y,axis=1)
                # construct randomly permuted minibatches
            #ind_global = np.concatenate(ind_global, axis=0)
            trainx = []
            trainy = []
            Pos = []
            Neg = []
            for t in range(int(X_train.shape[0]/(txs.shape[0]))):
                possible_classes = np.unique(tys);
                txs_pos = np.zeros(np.shape(txs)).astype(theano.config.floatX)
                txs_neg = np.zeros(np.shape(txs)).astype(theano.config.floatX)
                for j in range(0,txs.shape[0]):
                    label = tys[j]
                    ind_pos = ind_global[j,tys[ind_global[j]]==label];
                    ind_neg = ind_global[j,tys[ind_global[j]]!=label];
                    txs_pos[j,:] = txs[ind_pos[np.random.randint(len(ind_pos), size=1)[0]],:]
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
            l_sum = 0.0
            l_rbm_sum = 0.0
            l_lab_sum = 0.0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train,trainx,Pos,Neg, 50, shuffle=True):
                inputs = batch[0]
                l,l_rbm,l_lab = train_fn(inputs,batch[1],batch[2],batch[3])
                #print(train_re)
                #print(l)
                l_sum = l_sum + l
                l_rbm_sum = l_rbm_sum + l_rbm
                l_lab_sum = l_lab_sum + l_lab
                train_batches += 1
            loss_total.append(l_sum/train_batches)
            loss_unsup.append(l_rbm_sum/train_batches)
            loss_sup.append(l_lab_sum/train_batches)
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s, total loss: {} and rbm loss: {} and triplet loss: {}".format(
                epoch + 1, num_epochs, time.time() - start_time, loss_total[-1], loss_unsup[-1],loss_sup[-1]))
        with open(model_name, 'w') as f:
            cPickle.dump([rbm], f)

    #rbm = cPickle.load('rbm_2.pkl')

    '''if(for_training):
        x_value = np.arange(1, num_epochs+1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Error')
        plt.title("MNIST")
        #plt.plot(x_value, trainErrors, 'b-', label='train')
        plt.plot(x_value, valRE, 'r-', label='validation')
        plt.plot(x_value, trainRE, 'b-', label='train')
        plt.grid(b=None, which='major', axis='both')
        plt.legend(loc='upper right')
        plt.show()
'''
    def get_class(fTrain,ftemp,yTrain):

        ftemp = np.tile(ftemp, (fTrain.shape[0],1))
        dist = np.sum((fTrain>0.5)!=(ftemp>0.5),axis=1)
        ind = np.argsort(dist)

        return yTrain[ind]

    '''data = np.load('mnist.npz')
    testy = data['y_test'].astype(np.int32)'''
    testx, testy = cifar10_data.load('/home/maciej/Pobrane/cifar-10-batches-py', subset='test')
    trainx, trainy = cifar10_data.load('/home/maciej/Pobrane/cifar-10-batches-py', subset='train')
    testx, testy = cifar10_data.load('/home/maciej/Pobrane/cifar-10-batches-py', subset='test')
    trainx, trainy = cifar10_data.load('/home/maciej/Pobrane/cifar-10-batches-py', subset='train')


    test_features = samFun(X_test)
    test_features = test_features[0]
    train_features = samFun(X_train_out)
    train_features = train_features[0]
    prec = 0;
    test_features[test_features>0.5] = 1;
    test_features[test_features<0.5] = 0;
    train_features[train_features>0.5] = 1;
    train_features[train_features<0.5] = 0;
    '''print(test_features[20:40,:])
    print(testy[0:20])
    print(train_features[20:40,:])
    print(trainy[0:20])'''
    '''print('Retrieval results on test data')
    for k in range(np.shape(test_features)[0]):
        ftemp = test_features[k,:]
        class_values = get_class(np.delete(test_features,k,0),ftemp,np.delete(testy,k,0))
        y_true = (testy[k] == class_values)
        y_scores = np.arange(y_true.shape[0],0,-1)
        ap = average_precision_score(y_true, y_scores)
        prec = prec + ap
    prec = prec/float(np.shape(test_features)[0])
    print(prec)
    prec = 0;
    print('Retrieval results on train data')
    for k in range(np.shape(test_features)[0]):
        ftemp = test_features[k,:]
        class_values = get_class(train_features,ftemp,trainy)
        y_true = (testy[k] == class_values)
        y_scores = np.arange(y_true.shape[0],0,-1)
        ap = average_precision_score(y_true, y_scores)
        prec = prec + ap
    prec = prec/float(np.shape(test_features)[0])
    print(prec)'''

    rng_test = np.random.RandomState(10)
    inds = rng_test.permutation(test_features.shape[0])
    test_features = test_features[inds]
    testy = testy[inds]
    X_test = X_test_ver[inds]

    X_test[X_test>0.5] = 1;
    X_test[X_test<0.5] = 0;
    txs = []
    tys = []
    txs_before = []
    for j in range(10):
        txs.append(test_features[testy==j][:1000])
        tys.append(testy[testy==j][:1000])
        txs_before.append(X_test[testy==j][:1000])
    '''for j in range(10):
        txs.append(trainx[trainy==j])
        tys.append(trainy[trainy==j])'''
    X_test = np.concatenate(txs_before, axis=0)
    test_features = np.concatenate(txs, axis=0)
    testy = np.concatenate(tys, axis=0)

    X_train_ver[X_train_ver>0.5] = 1;
    X_train_ver[X_train_ver<0.5] = 0;
    '''Y = cdist(X_test,X_train_ver,'hamming')

    ind = np.argsort(Y,axis=1)
    prec = 0.0;
    acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];

    mAP_per_class = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    # calculating statistics
    for k in range(np.shape(test_features)[0]):
        class_values = trainy[ind[k,:]]
        y_true = (testy[k] == class_values)
        y_scores = np.arange(y_true.shape[0],0,-1)
        #y_scores = Y[k,ind[k,:]][::-1]
        y_true = (testy[k] == trainy)
        y_scores = 1 - Y[k,:]/float(max(Y[k,:]))
        ap = average_precision_score(y_true, y_scores)
        #ap = average_precision_score(y_true, y_scores)
        mAP_per_class[testy[k]] = mAP_per_class[testy[k]] + ap;
        prec = prec + ap
        for n in range(len(acc)):
            a = class_values[0:(n+1)]
            counts = np.bincount(a)
            b = np.where(counts==np.max(counts))[0]
            if testy[k] in b:
                acc[n] = acc[n] + (1.0/float(len(b)))
    prec = prec/float(np.shape(test_features)[0])
    mAP_per_class = [10.0*x / float(np.shape(test_features)[0])  for x in mAP_per_class]
    print(mAP_per_class)
    acc= [x / float(np.shape(test_features)[0]) for x in acc]
    print("Final results: ")
    print("mAP value: %.4f "% prec)
    for k in range(len(acc)):
        print("Accuracy for %d - NN: %.2f %%" % (k+1,100*acc[k]) )
    '''
    Y = cdist(test_features,train_features,'hamming')

    ind = np.argsort(Y,axis=1)
    prec = 0.0;
    acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];

    mAP_per_class = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    # calculating statistics
    for k in range(np.shape(test_features)[0]):
        class_values = trainy[ind[k,:]]
        y_true = (testy[k] == class_values)
        y_scores = np.arange(y_true.shape[0],0,-1)
        #y_scores = Y[k,ind[k,:]][::-1]
        y_true = (testy[k] == trainy)
        y_scores = Y[k,:]
        ap = average_precision_score(y_true, y_scores)
        '''ap = average_precision_score(y_true[0:100], y_scores[0:100])
        if(np.isnan(ap)==False):
            prec = prec + ap
        '''
        #ap = average_precision_score(y_true, y_scores)
        mAP_per_class[testy[k]] = mAP_per_class[testy[k]] + ap;
        prec = prec + ap
        for n in range(len(acc)):
            a = class_values[0:(n+1)]
            counts = np.bincount(a)
            b = np.where(counts==np.max(counts))[0]
            if testy[k] in b:
                acc[n] = acc[n] + (1.0/float(len(b)))
    prec = prec/float(np.shape(test_features)[0])
    mAP_per_class = [10.0*x / float(np.shape(test_features)[0])  for x in mAP_per_class]
    print(mAP_per_class)
    acc= [x / float(np.shape(test_features)[0]) for x in acc]
    print("Final results: ")
    print("mAP value: %.4f "% prec)
    for k in range(len(acc)):
        print("Accuracy for %d - NN: %.2f %%" % (k+1,100*acc[k]) )


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a restricted Boltzmann machine on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'rbm' for a Restricted Boltzmann Machine (RBM),")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)