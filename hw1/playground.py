from load_data import DataGenerator
import torch
import numpy as np
from hw1 import MANN

data_generator = DataGenerator(5, 1, device=torch.device('cpu'))
images, labels = data_generator.sample_batch('train',1)


