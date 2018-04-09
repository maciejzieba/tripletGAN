import argparse


def settings_binary():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1)
    parser.add_argument('--seed_data', default=1)
    parser.add_argument('--count', default=500)
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--unlabeled_weight', type=float, default=1.)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--use_pretrained', default=True)
    parser.add_argument('--generator_pretrained', default='gen_params_triplet_binary_pretrained.npz')
    parser.add_argument('--discriminator_pretrained', default='disc_params_triplet_binary_pretrained.npz')
    parser.add_argument('--generator_out', default='gen_params_triplet_binary.npz')
    parser.add_argument('--discriminator_out', default='disc_params_triplet_binary.npz')
    parser.add_argument('--data_dir', type=str, default='/home/maciej/Pobrane/cifar-10-batches-py')
    # Possible values 'difficult_first', 'easy_first', 'random'
    parser.add_argument('--triplet_positives_ordering', type=str, default='difficult_first')
    parser.add_argument('--triplet_negatives_ordering', type=str, default='random')
    args = parser.parse_args()
    return args
