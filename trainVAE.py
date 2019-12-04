import numpy as np
from astroML.datasets import sdss_corrected_spectra
import torch.utils.data
from torch import optim
from InfoVAE import InfoVAE
import os

tag = '64k_2par_20190703'
os.makedirs(tag)

data = np.load('spec64k.npz')
rawspec = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)
spec = rawspec / data['norms'][:,None] # normalize spectra with PCA norm
spec_err  = data['spec_err'] / data['norms'][:,None]
lowspecerr = (data['spec_err'] < 0.25) # some errors ~ 0 or are negative?
spec_weig = 1./(spec_err*spec_err + 1./(2e6)) # soften largest weights
meanspec = np.mean(spec, axis=0)
mask = data['mask']
spec_weig[mask] = 0 # if spectrum masked, act as if err = inf
spec_weig[lowspecerr] = 0

print('median weight %0.2e' % (np.median(spec_weig[spec_weig > 0])))
print('maximum weight %0.2e' % (np.amax(spec_weig)))

# only train and validate on spectroscopically confirmed galaxies/QSOs
spec_cln = data['lineindex_cln']
specmask = (spec_cln > 1) & (spec_cln < 7)

# deterministic outlier inclusion in training set
np.random.seed(20190605)
outlierfrac = 0.000
noutlier = min(int(spec.shape[0] * outlierfrac), np.sum(specmask == False))
outlieridx = np.random.choice(np.where(specmask == False)[0], size=noutlier, replace=False)
specmasktrain = (spec_cln > 1) & (spec_cln < 7)
specmasktrain[outlieridx] = True
# deterministic class exclusion in training set
excludefrac = 0.000
excludeclas = 4
nexclude = int(np.sum(spec_cln == excludeclas) * excludefrac)
excludeidx = np.random.choice(np.where(spec_cln == excludeclas)[0], size=nexclude, replace=False)
specmasktrain[excludeidx] = False

trainfrac = 0.75
ntrain = int(spec.shape[0] * trainfrac)
nvalid = spec.shape[0] - ntrain
nfeat = spec.shape[1]
np.random.seed(20190425) # make validation set deterministic
permutation = np.random.permutation(spec.shape[0])
np.random.seed()
trainidx = permutation[0:ntrain]
trainidx = trainidx[np.where(specmasktrain[trainidx])[0]]
valididx = permutation[-1-nvalid:-1]
valididx = valididx[np.where(specmask[valididx])[0]]
trainspec = spec[trainidx,:]
validspec = spec[valididx,:]
trainmask = mask[trainidx,:].astype(np.float32)
validmask = mask[valididx,:].astype(np.float32)
trainspecweig = spec_weig[trainidx,:]
validspecweig = spec_weig[valididx,:]
np.savez(tag+'/datasplit.npz', trainidx=trainidx, valididx=valididx)

torch.set_num_threads(4)

batch_size=64
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(trainspec-meanspec, dtype=torch.float32), torch.tensor(trainspecweig, dtype=torch.float32)),
    batch_size=batch_size,
    shuffle=True)
valdloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(validspec-meanspec, dtype=torch.float32), torch.tensor(validspecweig, dtype=torch.float32)),
    batch_size=nvalid)

def train(): #model, optimizer, epoch, min_valid_loss, badepochs
    model.train()
    train_loss = 0
    train_logL = 0
    for batch_idx, data in enumerate(dataloader):
        spectrum, weig = data
        optimizer.zero_grad()
        loss, logL, KLD, MMD = model.loss(spectrum, weig, epoch)
        loss.backward()
        train_loss += loss.item()
        train_logL += logL.item()
        optimizer.step()
    train_loss /= len(dataloader.dataset)
    
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        valid_logL = 0
        valid_KLD = 0
        valid_MMD = 0

        for valddata in valdloader:
            spectrum, weig = valddata
            loss, logL, KLD, MMD = model.loss(spectrum, weig, epoch)
            valid_loss += loss.item()
            valid_logL += logL.item()
            valid_KLD += KLD.item()
            valid_MMD += MMD.item()
        
        valid_loss /= len(valdloader.dataset)
        valid_logL /= -len(valdloader.dataset)
        valid_KLD  /= len(valdloader.dataset)
        valid_MMD  /= len(valdloader.dataset)
    return valid_loss, valid_logL, valid_KLD, valid_MMD

class EarlyStopper:
    def __init__(self, precision=1e-3, patience=10):
        self.precision = precision
        self.patience = patience
        self.badepochs = 0
        self.min_valid_loss = float('inf')
        
    def step(self, valid_loss):
        if valid_loss < self.min_valid_loss*(1-self.precision):
            self.badepochs = 0
            self.min_valid_loss = valid_loss
        else:
            self.badepochs += 1
        return not (self.badepochs == self.patience)

epochs = 200
log_interval = 10
mdl_ncode = 2
n_config = 100

mdl_MSE = np.zeros(n_config)
mdl_KLD = np.zeros(n_config)
mdl_MMD = np.zeros(n_config)

for config in range(n_config):
    alpha = 0
    lambd = np.exp(np.random.uniform(0, np.log(1e5)))
    dropout = 0#0.9*np.random.uniform()
    dfac = 1./(1.-dropout)
    nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(dfac*mdl_ncode+1), np.log(dfac*2*nfeat)))))
    nhidden2 = int(np.ceil(np.exp(np.random.uniform(np.log(dfac*mdl_ncode+1), np.log(nhidden)))))
    print('config %i, alpha = %0.1f, lambda = %0.1f, dropout = %0.2f; 2 hidden layers with %i, %i nodes' % (config, alpha, lambd, dropout, nhidden, nhidden2))
    model = InfoVAE(alpha=alpha, lambd=lambd, nfeat=nfeat, nhidden=nhidden, nhidden2=nhidden2, ncode=mdl_ncode, dropout=dropout)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    stopper = EarlyStopper(patience=10)

    for epoch in range(1, epochs + 1):
        valid_loss, valid_logL, valid_KLD, valid_MMD = train()
        if epoch % log_interval == 0:
            print('====> Epoch: {} VALIDATION Loss: {:.2e} logL: {:.2e} KL: {:.2e} MMD: {:.2e}'.format(
                  epoch, valid_loss, valid_logL, valid_KLD, valid_MMD))

        scheduler.step(valid_loss)
        if (not stopper.step(valid_loss)) or (epoch == epochs):
            print('Stopping')
            print('====> Epoch: {} VALIDATION Loss: {:.2e} logL: {:.2e} KL: {:.2e} MMD: {:.2e}'.format(
                  epoch, valid_loss, valid_logL, valid_KLD, valid_MMD))
            model.MSE = -valid_logL
            model.KLD = valid_KLD
            model.MMD = valid_MMD
            mdl_MSE[config] = model.MSE
            mdl_KLD[config] = model.KLD
            mdl_MMD[config] = model.MMD
            torch.save(model, tag+'/%04i.pth' % config)
            break

np.savez(tag+'/metrics.npz', MSE=mdl_MSE, KLD=mdl_KLD, MMD=mdl_MMD)
