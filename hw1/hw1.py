import argparse
import os
import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()
        
        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
    
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = torch.nn.LSTM(num_classes + input_size, 
                                    model_size, 
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        
        self.dnc = DNC(
                       input_size=num_classes + input_size,
                       output_size=num_classes,
                       hidden_size=model_size,
                       rnn_type='lstm',
                       num_layers=1,
                       num_hidden_layers=1,
                       nr_cells=num_classes,
                       cell_size=64,
                       read_heads=1,
                       batch_first=True,
                       gpu_id=-1,
                       )

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images
            
            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:
            
            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #detatch the original query labels to copy back in so that you do not lose the information
        query_labels = input_labels[:,-1,:,:].detach().clone()
        #fill the k+1th example with zeros for labels
        for b in np.arange(input_labels.shape[0]):
            for n in np.arange(input_labels.shape[2]):
                input_labels[b,-1,n,:] = torch.zeros(input_labels.shape[3])
        
        #concatentate images and labels along 3rd axis
        images_and_labels = torch.cat((input_images,input_labels),dim=3)
        
        #flatten 1st and 2nd dimension in the RIGHT ORDER
        input_tensor = torch.flatten(images_and_labels,start_dim=1,end_dim=2)

        #pass through both LSTMs
        out, _ = self.layer1(input_tensor.float())
        out, _ = self.layer2(out)
        out = out.reshape(out.shape[0], self.samples_per_class + 1, self.num_classes, out.shape[2])
        input_labels[:,-1,:,:] = query_labels
        return out 

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs
            
            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels
                
        Returns:
            scalar loss
        """

        preds_query = preds[:,-1,:,:]
        preds_query = preds_query.reshape(preds_query.shape[0] * preds_query.shape[1],preds_query.shape[2]) #Bx1, N
        labels_query = labels[:,-1,:,:]
        labels_query = labels_query.reshape(labels_query.shape[0] * labels_query.shape[1],labels_query.shape[2]) #Bx1, N
        labels_query = torch.argmax(labels_query, dim=1)
        return F.cross_entropy(preds_query, labels_query) 

def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cuda")
    writer = SummaryWriter(config.logdir)

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes, 
                                   config.num_samples, 
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples, 
                 model_size=config.model_size)
    
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1, 
                                        config.num_samples + 1, 
                                        config.num_classes, 
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)

            print(pred.eq(labels).double().mean().item())
            
            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy', 
                              pred.eq(labels).double().mean().item(),
                              step)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str, 
                        default='run/log')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())