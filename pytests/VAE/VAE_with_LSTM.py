from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
import os
import sys
import math
from rdkit.Chem import AllChem as Chem
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import socket
import logging
import datetime
from tqdm import tqdm
start_time = time.time()



log = logging.getLogger(__name__)

'''
# Get MPI:
try:
    from mpi4py import MPI

    WITH_DDP = True
    #LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    #device = 'gpu' if WITH_CUDA else 'CPU'
    #device = torch.device(f'cuda:{LOCAL_RANK}') if torch.cuda.is_available() else torch.device('cpu')
    #print(f'Using device: {device}, Local Rank: {LOCAL_RANK}')

    LOCAL_RANK = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f'cuda:{LOCAL_RANK}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Local Rank: {LOCAL_RANK}")

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)

    # Set up process group
    def init_process_group(rank, world_size, backend=None):
        if WITH_CUDA:
            backend = 'nccl' if backend is None else str(backend)
        else:
            backend = 'gloo' if backend is None else str(backend)

        dist.init_process_group(
            backend,
            rank=int(rank),
            world_size=int(world_size),
            init_method='env://',
        )

    # Broadcast MASTER_ADDR from rank 0 to other processes
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

    init_process_group(RANK, SIZE)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)
'''

WITH_DDP = True
#LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()

WITH_CUDA = torch.cuda.is_available()
#device = 'gpu' if WITH_CUDA else 'CPU'
#device = torch.device(f'cuda:{LOCAL_RANK}') if torch.cuda.is_available() else torch.device('cpu')
#print(f'Using device: {device}, Local Rank: {LOCAL_RANK}')

# LOCAL_RANK = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
LOCAL_RANK = RANK % torch.cuda.device_count()
torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f'cuda:{LOCAL_RANK}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, Local Rank: {LOCAL_RANK}")
def progress(it, desc="ITER"):
    if RANK==0:
        return tqdm(it, desc)
    else:
        return it

def print_rank_0(s):
    if RANK==0:
        print(s, flush=True)

# pytorch will look for these
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)

# Set up process group
def init_process_group(rank, world_size, backend=None):
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    print(os.environ['RANK'])
    print(os.environ['WORLD_SIZE'])
    print_rank_0(os.environ['MASTER_ADDR'])
    print_rank_0(os.environ['MASTER_PORT'])

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
        timeout=datetime.timedelta(seconds=300)
    )

# Broadcast MASTER_ADDR from rank 0 to other processes
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = str(2345)

init_process_group(RANK, SIZE)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

# Load and preprocess data
with open('output_10.txt', 'r') as file:
    names = file.read().splitlines()

# Define a mapping for the unique characters to indices
char_to_index = {char: index for index, char in enumerate("CO=#()123456")}
num_unique_chars = len(char_to_index)
seq_length = 24

# Create a custom dataset
class NamesDataset(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        one_hot = np.zeros((seq_length, num_unique_chars), dtype=int)
        for i, char in enumerate(name):
            one_hot[i, char_to_index[char]] = 1
        return torch.tensor(one_hot, dtype=torch.float32).to(device)

# Create dataset and calculate sizes for train/test splits
dataset = NamesDataset(names)
test_size = int(0.05 * len(dataset))  # 5% of the dataset for testing
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DistributedSampler for train and test datasets
train_sampler = DistributedSampler(train_dataset, num_replicas=SIZE, rank=RANK, shuffle=True)
test_sampler = DistributedSampler(test_dataset, num_replicas=SIZE, rank=RANK, shuffle=False)

# Modify DataLoader to use DistributedSampler
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, sampler=test_sampler, drop_last=False)

# Number of samples across all datasets for reference
n_data = len(dataset)

# Define the functions
def log_density_gaussian(x, mu, logvar):
    # Calculate log density of Gaussian distribution
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density

def matrix_log_density_gaussian(x, mu, logvar):
    # Calculate log density of Gaussian distribution for each sample in the batch
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.full((batch_size, batch_size), 1 / M, dtype=torch.float32)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def mutual_information_loss(log_q_zCx, log_qz):
    # Compute mutual information loss
    return (log_q_zCx - log_qz).sum()

def total_correlation_loss(log_qz, log_prod_qzi):
    # Compute total correlation loss
    return (log_qz - log_prod_qzi).sum()

def dw_KL_loss_(log_prod_qzi, log_pz):
    # Compute total correlation loss
    return (log_prod_qzi - log_pz).sum()

def get_losses(latent_sample, latent_dist, n_data, is_mss=False):
    # Get log densities and log importance weight matrix
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss)
    
    # Compute MI and TC losses
    mi_loss = mutual_information_loss(log_q_zCx, log_qz)
    tc_loss = total_correlation_loss(log_qz, log_prod_qzi)
    dw_KL_loss = dw_KL_loss_(log_prod_qzi, log_pz)
    
    return mi_loss, tc_loss, dw_KL_loss

def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=False):
    batch_size, hidden_dim = latent_sample.shape

    # Calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # Calculate log p(z)
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # Use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

from rdkit import rdBase

# Disable RDKit error messages
rdBase.DisableLog('rdApp.error')

# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() > 0

# Build the Sequence-to-Sequence VAE model
class Seq2SeqVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, char_to_index, device):
        super(Seq2SeqVAE, self).__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        self.char_to_index = char_to_index

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc_encoder = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(100, latent_size)
        self.fc_logvar = nn.Linear(100, latent_size)

        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.decoder_lstm = nn.LSTM(100, hidden_size, num_layers=2, batch_first=True)
        #self.fc_hidden = nn.Linear(hidden_size, 200)
        #self.fc_output = nn.Linear(200, input_size)
        self.fc_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (h, c) = self.encoder_lstm(x)
        #h = h.squeeze(0)
        h = self.fc_encoder(h[1])
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = self.fc_decoder(z)
        #z = z.unsqueeze(1)
        ## Initialize the hidden and cell states for the LSTM
        #h0 = torch.zeros(4, z.size(0), self.hidden_size).to(z.device)
        #c0 = torch.zeros(4, z.size(0), self.hidden_size).to(z.device)
        #hidden = (h0, c0)

        #outputs = []
        #for _ in range(seq_len):
        #    # Use the previous output as the input for the next step
        #    output, hidden = self.decoder_lstm(z, hidden)
        #    output = self.fc_hidden(output)
        #    z = output
        #    output = self.fc_output(output)
        #    outputs.append(torch.sigmoid(output))

        #outputs = torch.stack(outputs, dim=1)
        #return outputs.squeeze(2)

        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder_lstm(z)
        output = self.fc_output(output)
        return torch.sigmoid(output)

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def hot_to_smiles_(self, hot_x):
        smiles = []
        for x in hot_x:
            temp_str = ""
            for j in x:
                index = torch.argmax(j).item()  
                if j[index] > 0.01:
                    char = list(self.char_to_index.keys())[index]  
                    temp_str += char
            smiles.append(temp_str)
        return smiles

    def hot_to_smiles(self, hot_x, strip=False):
        smiles = self.hot_to_smiles_(hot_x)
        if strip:
            smiles = [s.strip() for s in smiles]
        return smiles

    def balanced_parentheses(self, input_string):
        s = []
        balanced = True
        index = 0
        while index < len(input_string) and balanced:
            token = input_string[index]
            if token == "(":
                s.append(token)
            elif token == ")":
                if len(s) == 0:
                    balanced = False
                else:
                    s.pop()

            index += 1

        return balanced and len(s) == 0

    def matched_ring(self, s):
        return s.count('1') % 2 == 0 and s.count('2') % 2 == 0

    def fast_verify(self, s):
        return self.matched_ring(s) and self.balanced_parentheses(s)

    def smiles_to_mol(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            pass
        return None

    def smiles_distance_z(self, smiles, z0):
        x = torch.stack([NamesDataset([smile])[0] for smile in smiles])
        z_rep, _ = self.encode(x)
        return np.linalg.norm(z0 - z_rep.cpu().detach().numpy(), axis=1)

    def prep_mol_df(self, smiles, z):
        df = pd.DataFrame({'smiles': smiles})
        sort_df = pd.DataFrame(df[['smiles']].groupby(by='smiles').size().rename('count').reset_index())
        df = df.merge(sort_df, on='smiles')
        df.drop_duplicates(subset='smiles', inplace=True)
        df = df[df['smiles'].apply(self.fast_verify)]

        if len(df) > 0:
            df['mol'] = df['smiles'].apply(self.smiles_to_mol)
        if len(df) > 0:
            df = df[pd.notnull(df['mol'])]
        if len(df) > 0:
            df['distance'] = self.smiles_distance_z(df['smiles'], z)
            df['frequency'] = df['count'] / float(sum(df['count']))
            df = df[['smiles', 'distance', 'count', 'frequency']]
            df.sort_values(by='distance', inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def z_to_smiles(self, mu, logvar, z, device, decode_attempts=100, noise_norm=5.0, constant_norm=False, early_stop=None):
        Z = np.tile(z.cpu().detach().numpy(), (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(torch.tensor(Z, dtype=torch.float32).to(self.device), seq_length)

        #sampled_Z_list = []
        #for _ in range(decode_attempts):
        #    z = self.reparameterize(mu, logvar)        
        #    sampled_Z_list.append(z)
        #Z = torch.stack(sampled_Z_list).squeeze(dim=1)
        #X = self.decode(Z.to(torch.float32).to(self.device), seq_length)

        smiles = self.hot_to_smiles(X, strip=True)
        df = self.prep_mol_df(smiles, z.cpu().detach().numpy())
        return df

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, seq_length)

        # Sample SMILES only during evaluation
        sampled_smiles = None
        if not self.training:
            sampled_smiles = self.z_to_smiles(mu, logvar, z, torch.device(device))

        # Compute latent_sample, latent_dist, and n_data
        latent_sample = z
        latent_dist = (mu, logvar)
        
        # Compute losses
        mi_loss, tc_loss, dw_KL_loss = get_losses(latent_sample, latent_dist, n_data)
        
        return recon_x, mu, logvar, mi_loss, tc_loss, dw_KL_loss, sampled_smiles

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction Loss (BCE Loss)
        BCE = nn.BCELoss(reduction='sum')(recon_x, x)

        # KL Divergence Loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE, KLD

#betas = list(range(1, 11))  
betas = [1]
num_epochs = 2

# Create a directory for saving models
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
num_samples = 10

model = Seq2SeqVAE(input_size=num_unique_chars, hidden_size=128, latent_size=64, char_to_index=char_to_index, device=device).to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for beta in betas:

    # Print initial BCE and KLD before training begins
    total_BCE = 0.0
    total_KLD = 0.0
    total_MI = 0.0
    total_TC = 0.0
    total_dw_KL = 0.0

    for batch in progress(train_dataloader, desc="pre-train"):
        batch = batch.to(device)
        recon_batch, mu, logvar, mi_loss, tc_loss, dw_KL_loss, _ = model(batch)
        BCE, KLD = model.module.loss_function(recon_batch, batch, mu, logvar)
        total_BCE += BCE.item()
        total_KLD += KLD.item()
        total_MI += mi_loss.item()
        total_TC += tc_loss.item()
        total_dw_KL += dw_KL_loss.item()

    avg_BCE = total_BCE / len(train_dataloader.dataset)
    avg_KLD = total_KLD / len(train_dataloader.dataset)
    avg_MI = total_MI / len(train_dataloader.dataset)
    avg_TC = total_TC / len(train_dataloader.dataset)
    avg_dw_KL = total_dw_KL / len(train_dataloader.dataset)
    print_rank_0(f'Initial BCE: {avg_BCE}, Initial KLD: {avg_KLD}, Initial MI: {avg_MI}, Initial TC: {avg_TC}, Initial dw_KL: {avg_dw_KL}')

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        total_BCE = 0.0
        total_KLD = 0.0
        total_MI = 0.0
        total_TC = 0.0
        total_dw_KL = 0.0
        for batch in progress(train_dataloader, desc="train"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, mi_loss, tc_loss, dw_KL_loss, _ = model(batch)
            BCE, KLD = model.module.loss_function(recon_batch, batch, mu, logvar)
            #loss = BCE + (beta * KLD * epoch / 300)
            loss = BCE + (beta * KLD * epoch / 400) if (beta * epoch / 400) < 0.25 else BCE + (0.25 * KLD)
            #loss = BCE + (beta * (mi_loss + (1*tc_loss) + dw_KL_loss) * epoch / 300)
            loss.backward()
            optimizer.step()
            total_BCE += BCE.item()
            total_KLD += KLD.item()
            total_MI += mi_loss.item()
            total_TC += tc_loss.item()
            total_dw_KL += dw_KL_loss.item()

        # Save the model after each epoch
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        #torch.save(model.state_dict(), model_save_path)

        # Sample and save SMILES after the last epoch
        sampled_smiles_file = os.path.join(save_dir, f'sampled_smiles_epoch_{epoch+1}.txt')

        with open(sampled_smiles_file, 'w') as f:
            for _ in range(num_samples):
                # Sample a random input from the dataset
                sample_idx = np.random.randint(len(dataset))
                sample_data = dataset[sample_idx].unsqueeze(0).to(device)

                # Generate output from the model
                with torch.no_grad():
                    model.eval()  # Switch to evaluation mode
                    recon_sample, mu, logvar, _, _, _, sampled_smiles = model(sample_data)
                    model.train()  # Switch back to training mode

                # Convert one-hot encoding to SMILES
                max_values, max_indices = sample_data.squeeze().max(dim=1)
                max_values_list = max_values.tolist()
                max_indices_list = max_indices.tolist()
                input_smiles_indices = [max_indices_list[i] for i in range(len(max_values_list)) if max_values_list[i] > 0]
                input_smiles = ''.join([list(char_to_index.keys())[i] for i in input_smiles_indices])
                max_values, max_indices = recon_sample.squeeze().max(dim=1)
                max_values_list = max_values.tolist()
                max_indices_list = max_indices.tolist()
                output_smiles_indices = [max_indices_list[i] for i in range(len(max_values_list)) if max_values_list[i] > 0.001]
                output_smiles = ''.join([list(char_to_index.keys())[i] for i in output_smiles_indices])

                # Write to the file
                if RANK==0:
                    f.write(f'Input SMILES: {input_smiles}\n')
                    #f.write(f'Input SMILES: {sample_data}\n')
                    f.write(f'Output SMILES: {output_smiles}\n')
                    #f.write(f'Output SMILES: {recon_sample}\n')
                    #f.write(f'mu: {mu}\n')
                    #f.write(f'std: {torch.exp(0.5 * logvar)}\n')
                    #f.write(f'eps: {torch.randn_like(torch.exp(0.5 * logvar))}\n')
                    #f.write(f'std_eps: {torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)}\n')
                    f.write('Sampled SMILES:\n')
                    f.write(sampled_smiles.to_string(index=False) + '\n\n')


        if (epoch + 1) % 1 == 0:
	    # Print average BCE and KLD over the entire epoch for the current beta
            avg_BCE = total_BCE / len(train_dataloader.dataset)
            avg_KLD = total_KLD / len(train_dataloader.dataset)
            avg_MI = total_MI / len(train_dataloader.dataset)
            avg_TC = total_TC / len(train_dataloader.dataset)
            avg_dw_KL = total_dw_KL / len(train_dataloader.dataset)
            print_rank_0(f'Epoch {epoch + 1}/{num_epochs}, Beta: {beta}, BCE: {avg_BCE}, KLD: {avg_KLD}, MI: {avg_MI}, TC: {avg_TC}, dw_KL: {avg_dw_KL}')

        # Reconstruction accuracy
        total_tried_reconstruction = 0
        successful_reconstructions = 0

        # Validity
        total_decoded_sequences = 0
        valid_reconstructions = 0

        with torch.no_grad():
            model.eval()  # Switch to evaluation mode
            for batch in progress(test_dataloader, desc="validation"):
                batch = batch.to(device)
                mu, logvar = model.module.encode(batch)
                for _ in range(2):  # Repeat the process 10 times
                    z = model.module.reparameterize(mu, logvar)

                    # Reconstruction accuracy
                    recon_batch = model.module.decode(z, seq_len=seq_length)
                    recon_smiles = model.module.hot_to_smiles(recon_batch, strip=True)
                    input_smiles = model.module.hot_to_smiles(batch, strip=True)
                    total_tried_reconstruction += len(recon_smiles)
                    for recon_smile, input_smile in zip(recon_smiles, input_smiles):
                        if recon_smile == input_smile:
                            successful_reconstructions += 1

                # Validity
                for _ in range(100):
                    sampled_z = torch.FloatTensor(1, 64).normal_(0,1).to(device)
                    decoded_smiles_batch = model.module.hot_to_smiles(model.module.decode(sampled_z, seq_len=seq_length), strip=True)
                    total_decoded_sequences += len(decoded_smiles_batch)
                    for decoded_smiles in decoded_smiles_batch:
                        if is_valid_smiles(decoded_smiles):
                            valid_reconstructions += 1

            model.train()  # Switch to train mode

        # Calculate reconstruction accuracy and validity
        reconstruction_accuracy = successful_reconstructions / total_tried_reconstruction
        validity = valid_reconstructions / total_decoded_sequences

        print_rank_0(f"Reconstruction Accuracy: {reconstruction_accuracy}")
        print_rank_0(f"Validity: {validity}")


end_time = time.time()
total_time = end_time - start_time
print_rank_0(f'Total run time: {total_time} seconds')
