import os
from pathlib import Path
from collections import OrderedDict
import torch
import flwr as fl
import subprocess
from nnunet.utilities.io import (path_exists, read_json)
import sys
import pickle
import numpy as np

workdir_path = os.environ.get("WORKDIR_PATH")
task_name = os.environ.get("TASK_NAME")
nnunet_wrapper_path = os.environ.get("NNUNET_WRAPPER_PATH")
initial_model_dir = os.environ.get("INITIAL_MODEL_DIR")
initial_plan_dir = os.environ.get("INITIAL_PLAN_DIR")
initial_plan_identifier = os.environ.get("INITIAL_PLAN_IDENTIFIER")
folds = os.environ.get("FOLDS")
check_point = os.environ.get("CHECK_POINT")
trainer = os.environ.get("TRAINER")
PLANS = os.environ.get("PLANS")

def get_parameters(self):
    outdir = os.path.join(workdir_path, 'results/nnUNet/3d_fullres', task_name, f'{trainer}__{PLANS}', f'fold_{folds}')
    outdir_Path=Path(outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path_exists(outdir_Path) and any(outdir_Path.glob("*.model")):
        model_dir=os.path.join(outdir,check_point)
        model_checkpoint_data = torch.load(model_dir, map_location=device)
        model_checkpoint_weights = model_checkpoint_data['state_dict']
    else:
        model_checkpoint_data = torch.load(initial_model_dir, map_location=device)
        model_checkpoint_weights = model_checkpoint_data['state_dict']
    return [val.cpu().numpy() for _, val in model_checkpoint_weights.items()]
    
def set_parameters(self, parameters):
    outdir = os.path.join(workdir_path, 'results/nnUNet/3d_fullres', task_name, f'{trainer}__{PLANS}', f'fold_{folds}')
    outdir_Path=Path(outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path_exists(outdir_Path) and any(outdir_Path.glob("*.model")):
        model_dir=os.path.join(outdir,check_point)
        model_checkpoint_data = torch.load(model_dir, map_location=device)
        model_checkpoint_weights = model_checkpoint_data['state_dict']
        params_dict = zip(model_checkpoint_weights.keys(), parameters)
        updated_weights = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model_checkpoint_data['state_dict'] = updated_weights
        torch.save(model_checkpoint_data, model_dir)  
    else:
        model_checkpoint_data = torch.load(initial_model_dir, map_location=device)
        model_checkpoint_weights = model_checkpoint_data['state_dict']
        params_dict = zip(model_checkpoint_weights.keys(), parameters)
        updated_weights = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model_checkpoint_data['state_dict'] = updated_weights
        torch.save(model_checkpoint_data, initial_model_dir)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(self)

    def fit(self, parameters, config):
        set_parameters(self, parameters)
        arguments = ['plan_train', '--trainer', trainer, 
                     '--custom_split', f"{os.path.join(workdir_path, 'nnUNet_raw_data', task_name, 'splits.json')}",
                     '--overwrite_plans', initial_plan_dir,
                     '--overwrite_plans_identifier', initial_plan_identifier,
                     '--pretrained_weights', initial_model_dir,
                     '--fold', folds,
                     task_name, workdir_path]
        cmd = [sys.executable, nnunet_wrapper_path] + arguments
        print('------------------------------')
        print('------------------------------')
        print('[#] Running the train command: \n ', " ".join(arguments))
        print('------------------------------')
        print('------------------------------')
        subprocess.run(cmd)
        # extract the number of the training samples
        dataset_json_dir = os.path.join(workdir_path, 'nnUNet_raw_data', task_name, 'dataset.json')
        dataset_info = read_json(dataset_json_dir)
        return get_parameters(self), dataset_info['numTraining'], {}

    def evaluate(self, parameters, config):
        set_parameters(self, parameters)
        print('------------------------------')
        print('------------------------------')
        print('[#] Running the EVALUATION! \n ')
        print('------------------------------')
        print('------------------------------')
        outdir = os.path.join(workdir_path, 'results/nnUNet/3d_fullres', task_name, f'{trainer}__{PLANS}', f'fold_{folds}')
        outdir_Path=Path(outdir)
        if path_exists(outdir_Path) and any(outdir_Path.glob("*.model")):
            model_dir=os.path.join(outdir,check_point)
        else:
            model_dir = initial_model_dir
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_checkpoint_data = torch.load(model_dir, map_location=device)
        all_val_losses=model_checkpoint_data['plot_stuff'][1]
        mean_val_loss = np.mean(all_val_losses)
        loss = mean_val_loss

        # Get accuracy from the summary.json
        summary_file_dir = os.path.join(outdir,'validation_raw/summary.json')
        Data = read_json(summary_file_dir)
        accuracy = Data['results']['mean']['0']['Accuracy']

        # extract the number of the validation samples
        split_json_dir = os.path.join(workdir_path, 'nnUNet_raw_data', task_name, 'splits.json')
        splits_info = read_json(split_json_dir)
        existing_pickle_path=Path(workdir_path,'saving_all_round_chkpoints.pkl')
        if not path_exists(existing_pickle_path):
            existing_data={'Round': 0}
            with open(existing_pickle_path, "wb") as f:
                pickle.dump(existing_data, f)
            print('\n \n The pickle file for saving all round checkpionts is CREATED! \n \n ')
        # Load training and validation info
        plot_data=  model_checkpoint_data['plot_stuff'] 
        required_data= Data['results']
        with open(existing_pickle_path, "rb") as f:
            existing_data = pickle.load(f)
    
        current_round = existing_data['Round']+1
        existing_data[current_round]={'JSON_data': required_data, 'plot_stuff': plot_data}
        existing_data['Round'] = current_round
    
        with open(existing_pickle_path, "wb") as f:
            pickle.dump(existing_data, f)
        print('\n \n The pickle file for saving all round checkpionts is UPDATED! \n \n ')

        return float(loss), len(splits_info[int(folds)]['val']), {"accuracy": float(accuracy)} 

fl.client.start_numpy_client(server_address='localhost:8080', client=FlowerClient())
  