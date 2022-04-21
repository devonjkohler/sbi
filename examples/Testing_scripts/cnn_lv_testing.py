
import numpy as np
import pickle

## CNN functions
import torch
from torch.nn import Conv2d, ReLU, Linear, Sequential, MaxPool2d, AvgPool2d, Dropout, Module, CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

## SBI
import sbi.utils as utils

## Define model
def gillespie_simulator(propose_rates):
    t = 0.
    stop_time = 5000.
    s = torch.tensor([20., 40.])
    path = np.insert(s, 0, t, axis=0).reshape(1, 3)

    rate_functions = [lambda s: propose_rates[0] * s[0],
                      lambda s: propose_rates[1] * s[1] * s[0],
                      lambda s: propose_rates[2] * s[1]]
    n_func = len(rate_functions)

    transition_matrix = torch.tensor([[1, 0], [-1, 1], [0, -1]])

    for i in range(3001):

        sampling_weights = [f(s) for f in rate_functions]
        total_weight = sum(sampling_weights)

        probs = np.array([weight / total_weight for weight in sampling_weights])
        sample = np.random.choice(n_func, p=probs)
        t += np.random.exponential(1.0 / total_weight)

        s = s + transition_matrix[sample]
        s = torch.normal(s, .25)
        s[0] = max(1, s[0])
        s[1] = max(1, s[1])

        path = torch.cat((path, np.insert(s, 0, t, axis=0).reshape(1, 3)), axis=0)

    return path[2:].T

## Build CNN Model
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 6)),
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 6)),
            Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 6)),
            Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=(1, 6))
        )

        self.linear_layers = Sequential(
            Linear(384, 384)
        )

        self.relu = Sequential(
            ReLU(inplace=True),
            Dropout(p=.2)
        )

        self.output = Sequential(
            Linear(384, 3)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = self.relu(x)
        x = self.linear_layers(x)
        x = self.relu(x)
        x = self.linear_layers(x)
        x = self.relu(x)
        x = self.linear_layers(x)
        x = self.relu(x)

        x = self.output(x)

        return x

## Define class to batch data
class TraceDataSet(Dataset):

  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y

## CNN train function
def train(epoch, model, train_x, train_y, val_x, val_y,
          optimizer, criterion, train_losses, val_losses):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    # if torch.cuda.is_available():
    #     x_train = x_train.cuda()
    #     y_train = y_train.cuda()
    #     x_val = x_val.cuda()
    #     y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    # if epoch % 256 == 0:
        # printing the validation loss
    print('epoch : ', epoch + 1, '\t', 'val loss :', loss_val)
    print('epoch : ', epoch + 1, '\t', 'train loss :', loss_train)


def main():

    # Prior used to train nn (need it to span area for inference)
    prior = utils.BoxUniform(
        torch.tensor([0.001, 0.0001, 0.01]),
        torch.tensor([0.03, 0.01, 0.05])
    )

    obs_len = 3000

    # Sample 10000 traces
    obs_list = list()
    labels = list()
    for i in range(obs_len):
        prior_sample = prior.sample()
        labels.append(prior_sample)
        obs_list.append(gillespie_simulator(prior_sample))

    x = torch.stack(obs_list, axis=0)
    y = torch.stack(labels, axis=0)

    # Save observations
    print("trying to save cnn obs")
    with open(r'/scratch/kohler.d/code_output/biosim/cnn_lv_obs_smaller_prior.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("trying to save cnn labels")
    with open(r'/scratch/kohler.d/code_output/biosim/cnn_lv_labels_smaller_prior.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

    # print("loading data")
    # with open(r'/scratch/kohler.d/code_output/biosim/cnn_lv_obs.pickle', 'rb') as handle:
    #     x = pickle.load(handle)
    # with open(r'/scratch/kohler.d/code_output/biosim/cnn_lv_labels.pickle', 'rb') as handle:
    #     y = pickle.load(handle)
    # with open(r'../../../cnn_lv_obs.pickle', 'rb') as handle:
    #     x = pickle.load(handle)
    # with open(r'../../../cnn_lv_labels.pickle', 'rb') as handle:
    #     y = pickle.load(handle)
    print("data loaded")
    # x = x[:1500]
    # y = y[:1500]
    ## Prepare data
    v0_min = x[:, 0].min()
    v0_max = x[:, 0].max()
    x[:, 0] = (x[:, 0] - v0_min) / (v0_max - v0_min)

    v1_min = x[:, 1].min()
    v1_max = x[:, 1].max()
    x[:, 1] = (x[:, 1] - v1_min) / (v1_max - v1_min)

    v2_min = x[:, 2].min()
    v2_max = x[:, 2].max()
    x[:, 2] = (x[:, 2] - v2_min) / (v2_max - v2_min)

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.05)
    train_x = train_x.reshape(obs_len*.95, 1, 3, 3000)
    val_x = val_x.reshape(obs_len*.05, 1, 3, 3000)

    # defining the model
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.0005)
    criterion = CrossEntropyLoss()

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     criterion = criterion.cuda()

    # defining the number of epochs
    n_epochs = 30
    batch_size = 512
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []

    loader = DataLoader(TraceDataSet(train_x, train_y), batch_size=batch_size, shuffle=True)

    # training the model
    for epoch in range(n_epochs):
        print("epoch:{0}".format(str(epoch)))
        # loader = iter(loader)
        # for i in range(0, train_x.size()[0], batch_size):
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            # print("batch:{0}".format(str(i)))

            # batch_x, batch_y = loader.next()

            train(epoch, model, batch_x, batch_y, val_x,
                  val_y, optimizer, criterion, train_losses, val_losses)

    losses = {"train" : train_losses, "val" : val_losses}

    # print("trying to save cnn losses")
    # with open(r'../../../cnn_losses.pickle', 'wb') as handle:
    #     pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("trying to save cnn model")
    # with open(r'../../../cnn_model.pickle', 'wb') as handle:
    #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("saved")

    print("trying to save cnn losses")
    with open(r'/scratch/kohler.d/code_output/biosim/cnn_losses_smaller_prior.pickle', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("trying to save cnn model")
    # with open(r'/scratch/kohler.d/code_output/biosim/cnn_model.pickle', 'wb') as handle:
    #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(model.state_dict(), r'/scratch/kohler.d/code_output/biosim/cnn_model_smaller_prior.pth')
    print("saved")

if __name__ == '__main__':
    main()