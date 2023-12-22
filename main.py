import random

from model import FedAvgNetCIFAR
from server import *
import numpy as np
from torch import optim


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(19940817)
    random.seed(19940817)

    # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    _random_seeder(2022)
    model =  FedAvgNetCIFAR()

    # Optimization setups
    optimizer = optim.SGD(lr=0.01, momentum= 0.9, weight_decay= 1e-05)



    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    server = BaseServer(
        algo_params,
        model,
        data_distributed,
        optimizer,

        **args.train_setups.scenario,
    )

    return server
def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()

if __name__ == "__main__":
    # Load configuration from .json file
    opt = {'data_setups': {'root': './data', 'dataset_name': 'cifar10', 'batch_size': 50, 'n_clients': 100, 'partition': {'method': 'sharding', 'shard_per_user': 2}}, 'train_setups': {'algo': {'name': 'fedavg', 'params': {}}, 'scenario': {'n_rounds': 200, 'sample_ratio': 0.1, 'local_epochs': 5, 'device': 'cuda:0'}, 'model': {'name': 'fedavg_cifar', 'params': {}}, 'optimizer': {'params': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-05}}, 'scheduler': {'enabled': True, 'name': 'step', 'params': {'gamma': 0.99, 'step_size': 1}}, 'seed': 2022}, 'wandb_setups': {'project': 'NeurIPS2022', 'group': 'fedavg', 'name': 'fedavg'}}

    main(opt)


