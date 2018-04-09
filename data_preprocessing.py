import cifar10_data
import numpy as np
import theano.tensor as T
import lasagne.layers as ll
import theano as th


def get_train_data(data_dir, samples_per_class, seed_data):
    rng_data = np.random.RandomState(seed_data)
    trainx, trainy = cifar10_data.load(data_dir, subset='train')
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy==j][:samples_per_class])
        tys.append(trainy[trainy==j][:samples_per_class])

    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    return trainx, trainy, txs, tys

def get_test_data(data_dir, reduce_to=100, seed_data=1):
    rng_data = np.random.RandomState(seed_data)
    testx, testy = cifar10_data.load(data_dir, subset='test')
    if reduce_to:
        inds = rng_data.permutation(testx.shape[0])
        testx = testx[inds]
        testy = testy[inds]
        testx = testx[0:reduce_to]
        testy = testy[0:reduce_to]
    return testx, testy


def get_feature_extractor(model):
    x_temp = T.tensor4()
    features = ll.get_output(model[-1], x_temp, deterministic=True)
    return th.function(inputs=[x_temp], outputs=features)


def extract_features(model, input_data, batch_size=1):
    nr_batches = int(input_data.shape[0]/batch_size)
    extract = get_feature_extractor(model)
    for t in range(nr_batches):
        if t == 0:
            features = extract(input_data[t*batch_size:(t+1)*batch_size])
        else:
            features = np.concatenate((features, extract(input_data[t*batch_size:(t+1)*batch_size])),axis=0)
    return features
