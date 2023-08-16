import torch
torch.multiprocessing.set_start_method('spawn')

import numpy as np
SEED=12345
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

import h5py
import numpy as np
import time
import torch
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data as GraphData
from torch.utils.data import Dataset

batch_size = 64 
epoch = 100 

class ShowerDataset(Dataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    def __del__(self):

        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __len__(self):

        return self._entries

    def __getitem__(self, idx):

        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)

        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_features, group_ids, node_labels = node_info[:,:-3], node_info[:,-2].long(), node_info[:,-1].long()

        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22), dtype=torch.float32)
        edge_features, edge_index, edge_labels = edge_info[:,:-3], edge_info[:,-3:-1].long().t(), edge_info[:,-1].long()

        return GraphData(x = node_features,
                    	 edge_index = edge_index,
                    	 edge_attr = edge_features,
                    	 y = node_labels,
                    	 edge_label = edge_labels,
                    	 index = idx)

train_data = ShowerDataset(file_path = 'if-graph-train.h5')

from torch_geometric.loader import DataLoader
loader = DataLoader(train_data,
                    shuffle     = True,
                    num_workers = 0,
                    batch_size  = batch_size 
                    )

data = next(iter(loader))
data

import torch.nn as nn # Neural Network implementations in PyTorch

class EdgeLayer(nn.Module):
    def __init__(self, node_in, edge_in, edge_out, leakiness=0.0):
        super(EdgeLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.BatchNorm1d(2 * node_in + edge_in),
            nn.Linear(2 * node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

import torch.nn as nn # Neural Network implementations in PyTorch
from torch_scatter import scatter_mean # Fast computation of node group mean features

class NodeLayer(nn.Module):
    def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
        super(NodeLayer, self).__init__()

        self.node_mlp_1 = nn.Sequential(
            nn.BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1) # Aggregating neighboring node with connecting edges
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0)) # Building mean messages
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

import torch.nn as nn # Neural Network implementations in PyTorch
from torch_geometric.nn import MetaLayer

class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()

        # Initialize the updaters
        self.message_passing = torch.nn.ModuleList()

        # Update the node and edge feature N times (number of message passings, here = 3)
        node_input  = 16 # Number of input node features
        edge_input  = 19 # Number of input edge features
        node_output = 64  # Number of intermediate node features
        edge_output = 64  # Number of intermediate edge features
        leakiness = 0.1 # LeakyRELU activation leakiness
        self.num_mp = 3 # Number of message passings

        for i in range(self.num_mp):
            self.message_passing.append(
                MetaLayer(
                    edge_model = EdgeLayer(node_input, edge_input, edge_output, leakiness=leakiness),
                    node_model = NodeLayer(node_input, node_output, edge_output, leakiness=leakiness)
                )
            )
            node_input = node_output
            edge_input = edge_output


        # Reduce the number of node and edge features edge, as we are performing a simple classification
        self.node_predictor = nn.Linear(node_output, 2)
        self.edge_predictor = nn.Linear(edge_output, 2)

    def forward(self, data):

        # Loop over message passing steps, pass data through the updaters
        x = data.x
        e = data.edge_attr
        for i in range(self.num_mp):
            x, e, _ = self.message_passing[i](x, data.edge_index, e, batch=data.batch)

        # Reduce output features to 2 each
        x_pred = self.node_predictor(x)
        e_pred = self.edge_predictor(e)

        # Return
        res = {
            'node_pred': x_pred,
            'edge_pred': e_pred
            }

        return res

model = GNNModel()
model

output = model(data)
output.keys(), output['node_pred'].shape, output['edge_pred'].shape

lossfn = torch.nn.CrossEntropyLoss(reduction='mean') # Mean cross-entropy loss

node_loss = lossfn(output['node_pred'], data.y) # Primary identification loss
edge_loss = lossfn(output['edge_pred'], data.edge_label) # Edge classification loss

node_loss, edge_loss


def test_gnn_scores(model,loader,num_iterations=100,device='cuda'):

    from scipy.special import softmax

    node_event_v, edge_event_v     = [], [] # Stores the event ID of each node/edge
    node_pred_v, edge_pred_v       = [], [] # Stores the binary prediction for node and edges (argmax)
    node_softmax_v, edge_softmax_v = [], [] # Stores the softmax score for the primary and ON channel of node and edges, respectively
    node_label_v, edge_label_v     = [], [] # Stores the true binary label for node and edges
    edge_index_v = [] # Stores the edge index for each event

    with torch.set_grad_enabled(False):
        for data in loader:
            # Bring data to GPU, if requested
            if device != 'cpu':
                data = data.to(torch.device(device))

            prediction = model(data)

            node_pred, edge_pred = prediction['node_pred'].cpu().numpy(), prediction['edge_pred'].cpu().numpy()

            node_event_v.append(data.cpu().index[data.batch].numpy())
            edge_event_v.append(data.cpu().index[data.batch[data.edge_index[0]]].numpy())

            node_pred_v.append( np.argmax(node_pred, axis=1)    )
            edge_pred_v.append( np.argmax(edge_pred, axis=1)    )

            node_softmax_v.append( softmax(node_pred, axis=1)[:,1] )
            edge_softmax_v.append( softmax(edge_pred, axis=1)[:,1] )

            node_label_v.append( data.cpu().y )
            edge_label_v.append( data.cpu().edge_label )

            cids = np.concatenate([np.arange(c) for c in np.unique(data.batch.numpy(),return_counts=True)[1]])
            edge_index_v.append( cids[data.edge_index.numpy()].T )

            # Break if over the requested number of iterations
            #num_iterations -= 1
            #if num_iterations < 1:
            #    break

    return np.concatenate(node_event_v), np.concatenate(edge_event_v),\
           np.concatenate(node_pred_v), np.concatenate(edge_pred_v),\
           np.concatenate(node_softmax_v), np.concatenate(edge_softmax_v),\
           np.concatenate(node_label_v), np.concatenate(edge_label_v),\
           np.vstack(edge_index_v)

def train_gnn(loader, loaderTest, model, num_iterations=100, lr=0.001, optimizer='Adam', device='cpu'):

    # Create an optimizer
    optimizer = getattr(torch.optim,optimizer)(model.parameters(),lr=lr)

    # Now we run the training, and keep track of the loss values
    losses, edge_losses, node_losses = [], [], []
    loss_x, cur_loss_x, loss_step = [], 0, batch_size / 110409.
    node_accuracy, edge_accuracy, accuracy_epoch = [], [], []
    cur_epoch = num_iterations
    while cur_epoch > 0:

        batch = 1

        for data in loader:
            # Bring data to GPU, if requested
            if device != 'cpu':
                data = data.to(torch.device(device))

            # Prediction
            prediction = model(data)

            # Compute loss
            node_loss = lossfn(prediction['node_pred'], data.y) # Primary identification loss
            edge_loss = lossfn(prediction['edge_pred'], data.edge_label) # Edge classification loss

            loss = node_loss + edge_loss

            # Update model weights
            optimizer.zero_grad() # Clear gradients from previous steps
            loss.backward()       # Compute the derivative of the loss w.r.t. the model parameters using backprop
            optimizer.step()      # Steps the model weights according to the lr and gradients

            # Record loss
            losses.append(loss.item())
            node_losses.append(node_loss.item())
            edge_losses.append(edge_loss.item())
            cur_loss_x += loss_step
            loss_x.append(cur_loss_x)

            # Break if over the requested number of iterations
            #num_iterations -= 1
            #if num_iterations < 1:
            #    break
            print("batch: ", batch, " train total loss: ", losses[-1])
            batch += 1

        print("epoch: ", num_iterations - cur_epoch, " train total loss: ", losses[-1])
        cur_epoch -= 1

        node_event, edge_event, node_pred, edge_pred, node_softmax, edge_softmax, node_label, edge_label, edge_indices = test_gnn_scores(model, loaderTest, num_iterations=100)
        secondary_mask, primary_mask = node_label==0, node_label==1
        correct_node = np.sum(node_label[secondary_mask]==node_pred[secondary_mask])
        correct_node += np.sum(node_label[primary_mask]==node_pred[primary_mask])
        node_accuracy.append(correct_node / len(node_label))

        off_mask, on_mask = edge_label==0, edge_label==1
        correct_edge = np.sum(edge_label[off_mask]==edge_pred[off_mask])
        correct_edge += np.sum(edge_label[on_mask]==edge_pred[on_mask])
        edge_accuracy.append(correct_edge / len(edge_label))

        accuracy_epoch.append(cur_loss_x)

    return np.array(losses), np.array(node_losses), np.array(edge_losses), np.array(loss_x), np.array(node_accuracy), np.array(edge_accuracy), np.array(accuracy_epoch)

import torch
torch.manual_seed(123)

# Get the training set
dataset = ShowerDataset('if-graph-train.h5')

# Initialize the loader
loader = DataLoader(dataset = dataset,
                    shuffle = True,
                    batch_size = batch_size,
                    num_workers = 0)

datasetTest = ShowerDataset('if-graph-test.h5')

loaderTest = DataLoader(dataset = dataset,
                    shuffle = False,
                    batch_size = batch_size,
                    num_workers = 0,
                    pin_memory = False)

# Initialize the model
model = GNNModel().cuda()

tstart = time.time()
# Train!
losses, node_losses, edge_losses, loss_x, node_accuracy, edge_accuracy, accuracy_x = train_gnn(loader, loaderTest, model, num_iterations=epoch, device='cuda')
tend = time.time()
tLap = tend - tstart
h = int(tLap / 3600)
tLap = tLap % 3600
m = int(tLap / 60)
s = round(tLap % 60, 2)
print('Run took ', h, 'hour ', m, 'min ', s, 's')


from matplotlib import pyplot as plt

plt.figure(figsize = (12,8))
fig, ax1 = plt.subplots()

ax1.plot(loss_x, losses, 'b', label='Total loss')
ax1.plot(loss_x, node_losses, 'g', label='Node loss')
ax1.plot(loss_x, edge_losses, 'r',label='Edge loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc = 'upper left')

ax2 = ax1.twinx()
ax2.plot(accuracy_x, node_accuracy, "m.-", label='node accuracy')
ax2.plot(accuracy_x, edge_accuracy, "k.-", label='edge accuracy')
ax2.set_ylabel('accuracy')
ax2.set_ylim(0.93, 1)

ax2.legend(loc = 'upper right')
plt.grid()
plt.draw()
plt.savefig('loss_accuracy.png')

torch.save(model.state_dict(), "trained_model.w")
