# Dimensionality Reduction of SDSS Spectra with Variational Autoencoders
This is the code repository for Portillo, Parejko, Vergara, and Connolly (2020).

## Prerequisites
* Python 3
* PyTorch
* TensorFlow
* astroML 0.4
* numpy
* matplotlib
* scikit-learn

## Reproducing the figures
1. Download the SDSS spectra we used by running `download.sh`. The repository has pretrained (non-variational) autoencoders (AEs) and variational autoencoders (VAEs).
2. The Jupyter notebook `SDSS-VAE.ipynb` will produce all of the quantitative figures in the paper.

## Retraining the autoencoders
1. Download the SDSS spectra we used by running `download.sh`; alternately, the SDSS query, de-redshifting, and PCA infill can be rerun with `compute_sdss_pca.py`.
2. Run `trainVAE.py` to train a set of VAEs: this file can be edited to change the tag that the VAEs are saved with, the latent space dimension, the number of different VAEs trained, and the range of hyperparameters used, among other things.
3. The trained VAEs will be saved in a directory with the tag name, along with `metrics.npz` containing performance metrics.
4. Before training AEs, run `postprocess.py` to yield `spec64k_normed.npz`.
5. Run `wrapper.py` to train AEs. The latent space dimension can be specified with `--n_z`, the batch size can be specified with `--batch_size`, and the number of training epochs can be specified with `--epoch`.
