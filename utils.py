import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CEL = nn.CrossEntropyLoss()
CEL = CEL.to(device)

class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, batch_size, dim_in, dim_latent, dim_out):
        super().__init__()
        self.batch_size = batch_size
        self.dim_latent = dim_latent

        self.hidden_states = torch.zeros((1,batch_size, dim_latent)) # init hidden state to 0
        self.hidden_states = self.hidden_states.to(device)

        self.lin1 = nn.Linear(dim_in, dim_latent)
        self.lin1 = self.lin1.to(device)
        self.lin2 = nn.Linear(dim_latent, dim_latent)
        self.lin2 = self.lin2.to(device)
        self.lin3 = nn.Linear(dim_latent, dim_out)
        self.lin3 = self.lin3.to(device)

    def reinit_states(self): #reinit hidden states after each forward
        self.hidden_states = torch.zeros((1, self.batch_size, self.dim_latent))
        self.hidden_states = self.hidden_states.to(device)

    def one_step(self,x, h):
        return torch.tanh(self.lin1(x) + self.lin2(h)) # computes hidden state of state i of a batch of sequence

    def forward(self, batch):
        batch = batch.to(device)
        for i in range(batch.size(1)): # we compute all hidden states of the i th states of all the sequences in the batch
            new_hidden_state = self.one_step(batch[:,i], self.hidden_states[-1])
            self.hidden_states = torch.cat((self.hidden_states, new_hidden_state.reshape(1, self.batch_size, self.dim_latent)))

    def decode(self, h):
        return self.lin3(h) # got rid of tanh

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

# Test
def testing(data_test, batch_size, epoch, rnn, writer):
    predictions = [] # Liste de listes (liste de predictions pour chaque batch)
    tests = [] # Liste de listes (liste de targets pour chaque batch)
    loss_all_batches = 0
    for x,y in data_test:
        try:
            assert x.size(0) == batch_size
            y = y.to(device)
            tests.append(y)

            with torch.no_grad():
                rnn.forward(x)
                yhat = rnn.decode(rnn.hidden_states[-1])
                loss = CEL.forward(yhat, y)

            loss_all_batches += loss
            preds = torch.argmax(yhat, dim = 1) # Get predicted class
            predictions.append(preds)

            rnn.reinit_states()
        except AssertionError:
            print("assertion error")

    writer.add_scalar('Loss/test', loss_all_batches, epoch)
    print(f"Epoch {epoch}: loss {loss_all_batches}")

    # Score
    score = 0
    for i, preds in enumerate(predictions):
        for j, pred in enumerate(preds):
            if pred == tests[i][j]:
                score += 1

    score /= (len(predictions) * batch_size)
    writer.add_scalar('Precision/test', score, epoch)
    print(f"Epoch {epoch}: precision {score}")
    # print("TEST SCORE = ", score) # 0 - 1 error
