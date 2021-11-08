from utils import RNN, device,SampleMetroDataset
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
count = 0
epoch = 20 # 20 epochs, batch de 18 dure environe 4 mins et score de 53 %
for n_iter in range(epoch):
    for x,y in data_train:
        try:
            assert x.size(0) == BATCH_SIZE # last batch is of size 4
            rnn.forward(x)
            # MANY TO ONE ARCHITECTURE
            yhat = rnn.decode(rnn.hidden_states[-1])
            y = y.to(device)
            loss = CEL.forward(yhat, y)

            writer.add_scalar('Loss/train', loss, count)
            print(f"Itérations {count}: loss {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rnn.reinit_states()
            count +=1
        except AssertionError:
            continue

# Test
predictions = [] # Liste de listes (liste de predictions pour chaque batch)
tests = [] # Liste de listes (liste de targets pour chaque batch)
count = 0
for x,y in data_test:
    try:
        assert x.size(0) == BATCH_SIZE
        rnn.forward(x)
        yhat = rnn.decode(rnn.hidden_states[-1])
        preds = torch.argmax(yhat, dim = 1) # Get predicted class
        predictions.append(preds)
        y = y.to(device)
        tests.append(y)

        loss = CEL.forward(yhat, y)
        writer.add_scalar('Loss/test', loss, count)
        print(f"Itérations {count}: loss {loss}")
        count += 1

        rnn.reinit_states()
    except AssertionError:
        continue

# Score
score = 0
for i, preds in enumerate(predictions):
    for j, pred in enumerate(preds):
        if pred == tests[i][j]:
            score += 1

score /= (len(predictions) * BATCH_SIZE)
print("TEST SCORE = ", score) # 0 - 1 error
