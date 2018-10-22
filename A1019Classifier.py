import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import A1019load_data as dp

# Define parameters
parser = argparse.ArgumentParser(description = "Classifier")


# Model select 
parser.add_argument('--model_selection', default='lstm', help='choose the classifier model')
parser.add_argument('--continue_train', default=False, help='need to continue train the model')
parser.add_argument('--model_path', help='the path of the model')
# Define LSTM
parser.add_argument('--n_lstm', default=1, type=int,help='number of lstm')
parser.add_argument('--lstm_layer', default=1, type=int, help='number of lstm hidden layers')
parser.add_argument('--lstm_input', default=64, type=int, help='lstm hidden layer size')
parser.add_argument('--lstm_output', default=64, type=int, help='output size')
parser
# Define CNN
parser.add_argument('--1DConv', default=2, type=int, help ='number of 1D conv layers')
parser.add_argument('--2DConv', default=2, type=int, help ='number of 2D conv layers')

parser.add_argument('--n_class', default=55, type=int, help = 'number of classes')

# Training options
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--optimizer', default='Adam', help='Adam|SGD')
parser.add_argument('--learning_rate',default=0.0001, type=float)
parser.add_argument('--learning_rate_decay', default=0.5, type=float, help='lr decay factor')
parser.add_argument('--learning_rate_every', default=100, type=int, help='lr decay per')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')

# Dataset
parser.add_argument('--brain_signal_file', default='datasets/')

# Cuda
cuda = True if torch.cuda.is_available() else False

opt = parser.parse_args()
print(opt)

class LSTM_Model(nn.Module):
    
    def __init__(self, num_layers, input_size, hidden_size, output_size,n_class,n_lstm):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_class = n_class
        self.n_lstm = n_lstm
        # Define layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers=num_layers, batch_first=True)
        self.layer = nn.Sequential(nn.Linear(output_size,n_class),
                                   nn.Softmax(),
                                   )
    def forward(self, x):
        # initial 
        batch_size = x.size(0)
        lstm_init = [torch.zeros(self.num_layers,batch_size, self.output_size),torch.zeros(self.num_layers,batch_size, self.output_size)]
        if cuda :
            lstm_init = (lstm_init[0].cuda(),lstm_init[1].cuda())
        lstm_init = (Variable(lstm_init[0], requires_grad=x.requires_grad), Variable(lstm_init[1], requires_grad=x.requires_grad))
        
        # Forword LSTM
        for i in range(self.n_lstm):
            x = self.lstm(x, lstm_init)
        x = x[0][:,-1,:]
        x = self.layer(x)
        return x

# Prepare model
if opt.model_selection == 'lstm':
    model = LSTM_Model(opt.lstm_layer, opt.lstm_input, opt.lstm_input, opt.lstm_output, opt.n_class, opt.n_lstm)

optimizer = getattr(torch.optim, opt.optimizer)(model.parameters(), lr = opt.learning_rate)

if cuda: model.cuda()

# Prepare training data  
train=dp.load_data(opt.batch_size)      
        
        
# Training
for epoch in range(opt.epochs):
    # use dictionary to record the loss and accuracy
    ac = {'train':0, 'valid':0, 'test':0}
    losses = {'train':0, 'valid':0, 'test':0}
    count = {'train':0, 'valid':0, 'test':0}
    # Learning rate decay for SGD
    if opt.optimizer == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # training all batches
    for s in ("train","valid","test"):
        data=train[s]
        if s == 'train': 
            model.train()
        else:
            model.eval()
        
        # load data
        for _,batch_data in enumerate(data):
            # check CUDA
            
            input = torch.from_numpy(batch_data['x'])
            target = torch.from_numpy(batch_data['y'])
            if cuda:
                input = input.cuda(async = True)
                target = target.cuda(async = True)
            input = Variable(input, requires_grad = (s != "train")).float()
            target = Variable(target, requires_grad = (s!= "train")).long()
            # count size
            count[s] += input.data.size(0)
            # calculate loss and accuracy
            output = model(input)
            loss = F.cross_entropy(output, target)
            losses[s] += loss.item()
            
            _,pre = output.data.max(1)
            correct = pre.eq(target.data).sum()
            ac[s] += correct
            
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    # Print info at the end of the epoch
    content = "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch,
                                                                                                         losses["train"]/count["train"],
                                                                                                         ac["train"].item()/count["train"],
                                                                                                         losses["valid"]/count["valid"],
                                                                                                         ac["valid"].item()/count["valid"],
                                                                                                         losses["test"]/count["test"],ac["test"].item()/count["test"])
    print(content)
            
            
        
        
      
        
        
        
        