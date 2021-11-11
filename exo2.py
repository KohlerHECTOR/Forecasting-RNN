from utils import RNN, device,SampleMetroDataset, testing
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()
# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 18

DIM_LATENT = 16

LR = 1e-4

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
writer = SummaryWriter("runs/latent_dim_"+str(DIM_LATENT))



rnn = RNN(BATCH_SIZE, dim_in = DIM_INPUT, dim_latent = DIM_LATENT, dim_out = CLASSES)
optimizer = torch.optim.Adam(rnn.parameters())
CEL = torch.nn.CrossEntropyLoss()
CEL = CEL.to(device)


# Train
epoch = 40 # 20 epochs, batch de 18 dure environe 4 mins et score de 53 %
for n_iter in range(epoch):
    loss_all_batches = 0
    for x,y in data_train:
        try:
            assert x.size(0) == BATCH_SIZE # last batch is of size 4
            rnn.forward(x)
            # MANY TO ONE ARCHITECTURE
            yhat = rnn.decode(rnn.hidden_states[-1])
            y = y.to(device)
            loss = CEL.forward(yhat, y)
            loss_all_batches += loss



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rnn.reinit_states()

        except AssertionError:
            print("assertion error")

    writer.add_scalar('Loss/train', loss_all_batches, n_iter)
    print(f"Epoch {n_iter}: loss {loss_all_batches}")
    testing(data_test, BATCH_SIZE, n_iter, rnn, writer)
