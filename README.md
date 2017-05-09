# Triplet GAN 
[Maciej Zieba](https://www.ii.pwr.edu.pl/~zieba/), [Lei Wang](http://www.uow.edu.au/~leiw/)

This is the official code release for *Training Triplet Networks with GAN* ([arXiv](https://arxiv.org/abs/1704.02227)).

The code is based on *Improved Techniques with GAN* ([arXiv](https://arxiv.org/abs/1606.03498)) ([code](https://github.com/openai/improved-gan))

Please consider citing Training *Triplet Networks with GAN*:

@article{zieba2017training,
    title={Training Triplet Networks with GAN},
    author={Zieba, Maciej and Wang, Lei},
    journal={arXiv preprint arXiv:1704.02227},
    year={2017}
}

## Running the code

The training for MNIST data is performed by `tgan_mnist.py`.
The training for Cifar10 is performed by `tgan_cifar.py`.

The scripts train the triplet GAN on the considered dataset, serialize the discriminator and generator, and calculate  mAP and K-NN accuracy on test datasets using trained model.

The training code requires [Lasagne](http://lasagne.readthedocs.io/en/latest/). Using GPU is highly advised. The training procedure for Cifar10 requires 500 epochs of training to obtain results from the paper.
