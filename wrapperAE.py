import argparse
import autoencoderv3 as AE
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error

# In[2]:
"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of AE"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--n_z', type=int, default=2, help='Number neurons latent space', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--type_eval', type=str, default='training', choices=['training', 'validation'],
                        help='AE training or evaluate new data on previously trained AE')
    parser.add_argument('--method', type=str, default='transform', choices=['transform', 'generate', 'reconstruct'],
                        help='Evaluate data on AE method')
    return parser.parse_args()


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.type_eval == 'training':
        data = np.load('spec64k_normed.npz')
        X = data['spec'].copy()
        y = data['spec_class'].copy()
        wavelength = data['lam']

        X_train = X[data['trainidx']]
        y_train = y[data['trainidx']]
        X_test = X[data['valididx']]
        y_test = y[data['valididx']]

        modelAE = AE.Autoencoder(
            learning_rate=1e-3,
            batch_size=args.batch_size,
            training_epochs=args.epoch,
            display_step=5,
            n_z=args.n_z,
            n_hidden=[549, 110, 110, 549])
        modelAE.train(X_train)
        modelAE.save('LS_' + str(args.n_z))

    elif args.type_eval == 'validation':
        modelAE = AE.Autoencoder()
        modelAE.restore('LS_' + str(args.n_z))
        data = np.load('data2eval.npz')['arr_0']
        if args.method == 'transform': #Transform data by mapping it into the latent space.
            out = modelAE.transform(data)
        elif args.method == 'generate': #Generate data by sampling from latent space
            out = modelAE.generate(data)
        elif args.method == 'reconstruct': #Use AE to reconstruct given data
            out = modelAE.reconstruct(data)
        np.savez('data2eval_result.npz', out)




if __name__ == '__main__':
    main()
