from backends import get_discriminator_binary
from data_preprocessing import get_train_data, get_test_data, extract_features
from settings import settings_binary
from triplet_utils import load_model

args = settings_binary()
disc_layers = get_discriminator_binary()
load_model(disc_layers,args.discriminator_pretrained)

args = settings_binary()
trainx, trainy, txs, tys = get_train_data(args.data_dir, args.count, args.seed_data)
testx, testy = get_test_data(args.data_dir)

train_features = extract_features(disc_layers, trainx, args.batch_size)
test_features = extract_features(disc_layers, testx)
tx_features = extract_features(disc_layers, txs)


