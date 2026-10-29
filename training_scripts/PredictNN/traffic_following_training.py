# /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import os
import csv
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

class Model(nn.Module):
    def __init__(self, in_features = 3, h1 = 512, h2 = 512, h3 = 512, h4 = 512, h5 = 512, out_features = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4)
        # self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_features)
        self.dp = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.dp(x)
        x = F.sigmoid(self.fc2(x))
        x = self.dp(x)
        # x = F.sigmoid(self.fc3(x))
        # x = self.dp(x)
        # x = F.sigmoid(self.fc4(x))
        # x = self.dp(x)
        # x = F.sigmoid(self.fc5(x))
        # x = self.dp(x)
        x = self.out(x)
        return x
    
    def acc_loss(self, gt_acc, input):
        nn_output_acc = self.forward(input)
        tartget = torch.reshape(gt_acc, (nn_output_acc.shape))
        loss = nn.MSELoss()
        loss_accel = loss(nn_output_acc, tartget)
        return loss_accel

def data_loader(filename):
    input_state = []
    output_control = []
    with open(filename) as f:
        for points in csv.reader(f):
            ego_v = float(points[0])
            pv_ds_pred = float(points[3])
            pv_dv_pred = float(points[4])
            ego_a = float(points[5])
            
            input_state.append([ego_v, pv_dv_pred, pv_ds_pred])
            output_control.append(ego_a)
    return input_state, output_control

if __name__ == "__main__":
    # Import data and set it to tensors
    dirname = os.path.dirname(__file__)
    
    train_filename = dirname + '/train_pcc_driving_cycle.csv'
    test_filename = dirname + '/test_pcc_driving_cycle.csv'
    valid_filename = dirname + '/valid_pcc_driving_cycle.csv'
    
    train_x, train_y = data_loader(train_filename)
    test_x, test_y = data_loader(test_filename)
    valid_x, valid_y = data_loader(valid_filename)
    
    # Set numpy to torch
    train_x = torch.FloatTensor(train_x).cuda()
    train_y = torch.FloatTensor(train_y).cuda()
    
    test_x = torch.FloatTensor(test_x).cuda()
    test_y = torch.FloatTensor(test_y).cuda()
    
    with open(dirname + '/config/fcn_config.yaml', 'r') as f:
        fcn_config = yaml.safe_load(f)
        
    h1_neuron = fcn_config['fcn_hidden_layer']['h1']
    h2_neuron = fcn_config['fcn_hidden_layer']['h2']
    h3_neuron = fcn_config['fcn_hidden_layer']['h3']
    h4_neuron = fcn_config['fcn_hidden_layer']['h4']
    h5_neuron = fcn_config['fcn_hidden_layer']['h5']
    
    # Initialize model and optimizer
    model = Model(h1 = h1_neuron, h2 = h2_neuron, h3 = h3_neuron, h4 = h4_neuron, h5 = h5_neuron)
    #model.load_state_dict(torch.load(home_dir + '/ANL_ws/src/sumo_cra_traffic_sim/scripts/traffic_following_control.pt'))
    model.cuda()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.000001)
    epochs = 800
    batch_size = 512
    num_batch = int(len(train_y) / batch_size)
    losses_system = []
    loss = torch.tensor(0.)
    loss_record = []
    
    epoch_count = 0
    
    # Train the neural network
    for i in range(num_batch):
        for j in range(epochs):
            loss = model.acc_loss(train_y[batch_size * i: batch_size * (i + 1)], train_x[batch_size * i: batch_size * (i + 1), :])
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_count += 1
                loss_record.append(loss.tolist())
            print('The overall training percentage is: ' + str(round(i / num_batch * 100, 1)) + '%. Epoch of the ' + str(i) + 'th batch loss is ' + str(round(loss.tolist(), 3)), end="\r")
            
    print('\nTraining Finished.')
    plt.style.use('bmh')
    plt.figure(1)
    plt.plot(range(len(loss_record)), loss_record)
    plt.xlabel('Epoch')
    plt.ylabel('MSE LOSS')
    plt.show()

    # Test accuracy
    error_sum = 0
    with torch.no_grad():
        for i in range(len(test_y)):
            loss_test = model.acc_loss(test_y[i], test_x[i, :])

    # Save the model 
    torch.save(model.state_dict(), dirname + '/predicted_traffic_following_control_fcn_best.pt')