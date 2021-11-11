from utils import RNN, device,ForecastMetroDataset, testing_forecast
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()

CLASSES = 80 #max
LENGTH = 20
DIM_INPUT = 2
HORIZON = None
BATCH_SIZE = 1
DIM_LATENT = 16
LR = 6e-4
PATH = "data/"

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH) # x = st_1 : st_length - 1 ; y = st_2 : st_length
ds_test=ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

writer = SummaryWriter("runs/forecast/latent_dim_"+str(DIM_LATENT))

rnn = RNN(batch_size = CLASSES, dim_in = DIM_INPUT, dim_latent = DIM_LATENT, dim_out = DIM_INPUT)
optimizer = torch.optim.Adam(rnn.parameters())

MSE = torch.nn.MSELoss()
MSE = MSE.to(device)

# Train
epoch = 40 # 20 epochs, batch de 18 dure environe 4 mins et score de 53 %
for n_iter in range(epoch):
    loss_all_batches = 0
    for x,y in data_train:
        x = x.reshape((CLASSES, LENGTH - 1, DIM_INPUT))
        y = y.reshape((CLASSES, LENGTH - 1, DIM_INPUT))
        try:
            assert x.size(0) == CLASSES # last batch is of size 4
            rnn.forward(x)
            # MANY TO ONE ARCHITECTURE
            yhat = rnn.decode(rnn.hidden_states[1:])
            yhat = yhat.reshape((CLASSES, LENGTH - 1, DIM_INPUT))
            y = y.to(device)
            loss = MSE.forward(yhat, y)
            loss_all_batches += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rnn.reinit_states()

        except AssertionError:
            print("assertion error")

    writer.add_scalar('Loss/train', loss_all_batches, n_iter)
    print(f"Epoch {n_iter}: loss {loss_all_batches}")
    testing_forecast(data_test, BATCH_SIZE, n_iter, rnn, writer, MSE, HYPERPARAMS = (CLASSES, LENGTH, DIM_INPUT))
