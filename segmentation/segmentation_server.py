
import os
import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from collections import OrderedDict
import glob as glob
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,)
from flwr.server.client_proxy import ClientProxy
import argparse
import torch

def get_parameters():
    initial_model_dir="./initial_model.model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_checkpoint_data = torch.load(initial_model_dir, map_location=device)
    model_checkpoint_weights = model_checkpoint_data['state_dict']
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model_checkpoint_weights.items()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rounds",
        type=int,
        default=5,
        help="Number of rounds of federated learning (default: 5)",)
    parser.add_argument(
        "min_fit_clients",
        type=int,
        default=2,
        help="Then minimum number of clients used during training (default: 2)",)
    parser.add_argument(
        "min_aval_clients",
        type=int,
        default=2,
        help="The minimum number of clients that need to be connected to the server before a training round can start (default: 2)",)
    args = parser.parse_args()
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_aval_clients,
        initial_parameters=get_parameters(),)
    
    fl.server.start_server( 
        server_address='localhost:8080', 
        config=fl.server.ServerConfig(num_rounds=args.rounds), 
        strategy=strategy)

if __name__ == "__main__":
    main()