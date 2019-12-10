import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()

## yelp
parser.add_argument('--dataset', type=str, default='yelp', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--channel', type=int, default=8, help='number of channels in attention unit')
parser.add_argument('--L', type=int, default=3, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--att', type=bool, default=False, help='if use attention mechanism')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l1_weight', type=float, default=0.2, help='weight of kge regularization')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-5, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=1e-5, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=5, help='training interval of KGE task')


show_loss = False
show_topk = False

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)
