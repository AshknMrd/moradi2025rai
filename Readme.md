# Federate Learning for  Prostate Segmentation and Lesion Detection

**Optimizing Federated Learning Configurations for MRI Prostate Segmentation and Cancer Detection:**  

In this study, we proposed a federated learning (FL) approach for prostate segmentation on T2-weighted MRI data involving four clients and a total of 1,294 patient cases, as well as for detecting clinically significant prostate cancer (csPCa) on biparametric MRI data (T2W, HBV, and ADC) using three clients and 1,500 patient cases in total (data from [PI-CAI (Prostate Imaging: Cancer AI) grand challenge](https://pi-cai.grand-challenge.org/)). A schematic of the implemented scenarios, including details on data distribution, is presented in the following, prostate segmentation on the left and csPCa detection on the right.

<p align="center">
  <img src="Figs/topology.png" width="700" alt="Description">
</p>

### Abstract:

**Purpose:** To investigate and optimize a federated learning (FL) framework across multiple clients for biparametric MRI prostate segmentation and clinically significant prostate cancer (csPCa) detection.

**Materials and Methods:** A retrospective study was conducted using Flower FL to train a nnU- Net-based architecture for MRI prostate segmentation and lesion detection with data from January 2010 to August 2021. This included training and optimizing local epochs, federated rounds, and aggregation strategies for FL-based prostate segmentation on T2-weighted MRIs (four clients, 1294 patients) and csPCa detection using biparametric MRIs (three clients, 1440 patients). Performance was evaluated on independent test sets using the Dice score for segmentation and receiver operating characteristic and precision-recall curve analyses for detection, with p-values for performance differences from permutation testing.

**Results:** The FL configurations were independently optimized for both tasks, showing improved performance at 1 epoch 300 rounds using FedMedian for prostate segmentation and 5 epochs 200 rounds using FedAdagrad, for csPCa detection. Although the optimized FL model significantly improved performance and generalizability on independent test set compared with the average of local models, segmentation Dice score from 0.73±0.06 to 0.88±0.03 (P ≤ 0.01) and the lesion detection score, (AUC+AP)/2, from 0.63±0.07 to 0.74±0.06 (P ≤ 0.01), there was no proof of improved segmentation performance compared with the FL-baseline model.

**Conclusion:** FL enhanced the performance and generalizability of MRI prostate segmentation and csPCa detection compared with local models and optimizing its configuration further improved the lesion detection score, from 0.72±0.06 to 0.74±0.06 (P≤0.01).

#### Segmentation Performance:
<p align="center">
  <img src="Figs/Fig3.png" width="650" alt="Description">
</p>

#### Detection Performance:
<p align="center">
  <img src="Figs/Fig5.png" width="600" alt="Description">
</p>

## Installation:
Clone the entire repository to your local system, and then install the environment using the `.yml` file: 
```bash
git clone https://github.com/AshknMrd/moradi2025rai.git
conda env create -f environment.yml
```
Next, install the pipelines developed by the PI-CAI Grand Challenge organizers for data conversion (DICOM to MHA, MHA to nnU-Net Raw Data). Details for data conversion and evaluation for the Detection task, following the PI-CAI Grand Challenge guidelines, are provided in [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) and [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval), respectively. For the segmentation task, the Dice score, relative volume difference, and HD95 distance are used as evaluation metrics. The modified version of the nnUNet codebase ([version 1.7.0](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)) and Flower must be also installed:

```bash
cd nnUNet  
pip install -e .
pip install flwr==1.3.0
```

## Usage:
To reuse this repository, you must first create a working directory that includes the nnUNet folders, referred to as `workdir_dir`. Then, for each task, you can use the provided wrapper to run the nnUNet commands and the Flower-based script for the client and server. For FL configuration optimization, the number of federated rounds and local nnUNet training epochs can be set as desired and different server aggregation strategies can also be substituted in the server script using the following examples for setting the parameters.

```Python
strategy = FedAvg(fraction_fit=1.0,
              min_fit_clients=args.min_fit_clients,
              min_available_clients=args.min_aval_clients,
              initial_parameters=get_parameters(),)

strategy = FedAdagrad(fraction_fit=1.0,
              min_fit_clients=args.min_fit_clients,
              min_available_clients=args.min_aval_clients,
              initial_parameters=get_parameters(),
              eta=1e-1,eta_l=1e-1,tau=1e-9)

strategy = FedAdam(fraction_fit=1.0,
              min_fit_clients=args.min_fit_clients,
              min_available_clients=args.min_aval_clients,
              initial_parameters=get_parameters(),
              eta=1e-1,eta_l=1e-1,beta_1=0.9,
              beta_2=0.99,tau=1e-9)

strategy = FedMedian(fraction_fit=1.0,
              min_fit_clients=args.min_fit_clients,
              min_available_clients=args.min_aval_clients,
              initial_parameters=get_parameters())

strategy = FedYogi(fraction_fit=1.0,
              min_fit_clients=args.min_fit_clients,
              min_available_clients=args.min_aval_clients,
              initial_parameters=get_parameters(),
              eta=1e-2,eta_l=0.0316,beta_1=0.9,
              beta_2=0.99,tau=1e-3)
```

### Prostate Gland Segmentation Task: 
On one client side, run the local model for one epoch using the model plan and architecture determined by nnUNet. Of course, this requires the data to be preprocessed as detailed in the manuscript.

```Python
python train_pre_process_wrapper.py plan_train 
        --trainer 'nnUNetTrainerV2_Loss_Dice_FL' 
        --custom_split './workdir/nnUNet_raw_data/Task1001_segmentation_FL/splits.json' 
        --fold 0 
        'Task1001_segmentation_FL' './workdir'
```
Then after distributing the initial model and plan to all client sites, both the local and centralized models can be run as

 ```Python
python train_pre_process_wrapper.py plan_train 
        --trainer 'nnUNetTrainerV2_Loss_Dice_FL' 
        --custom_split './workdir/nnUNet_raw_data/Task1001_segmentation_FL/splits.json' 
        --overwrite_plans './initial_plan.pkl' 
        --overwrite_plans_identifier 'nnUNetData_plans_v2.1' 
        --pretrained_weights './initial_model.model' 
        --fold 0 
        'Task1001_segmentation_FL' './workdir'

 ```
and run the federated experiments by executing  `./server_run.sh` on the server side and `./client_run.sh` on each of the client sides.


### csPCa Detection Task:
Similarly, for the detection task, on one client side, run the local model for one epoch using the model plan and architecture determined by nnUNet (the data to be preprocessed as detailed in the manuscript).
```Python
python train_pre_process_wrapper.py plan_train 
        --trainer 'nnUNetTrainerV2_Loss_CE_FL' 
        --custom_split './workdir/nnUNet_raw_data/Task1001_detection_FL/splits.json' 
        --fold 0 
        'Task1001_detection_FL' './workdir'
```
and after distributing the initial model and plan to all client sites, both the local and centralized models can be run as

 ```Python
python train_pre_process_wrapper.py plan_train 
        --trainer 'nnUNetTrainerV2_Loss_CE_FL' 
        --custom_split './workdir/nnUNet_raw_data/Task1001_detection_FL/splits.json' 
        --overwrite_plans './initial_plan.pkl' 
        --overwrite_plans_identifier 'nnUNetData_plans_v2.1' 
        --pretrained_weights './initial_model.model' 
        --fold 0 
        'Task1001_detection_FL' './workdir'

 ```
and run the federated experiments by executing  `./server_run.sh` on the server side and `./client_run.sh` on each of the client sides.

### Evaluation: 
For each task, predictions can also be generated using the provided wrapper script as follows, and the evaluation metrics for the detection task can be computed using [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval). Please note that prediction and evaluation can be done directly using nnUNet also.

```Python
python train_pre_process_wrapper.py predict 
          --trainer 'trainer_name'
          --folds 0 
          --checkpoint 'model_final_checkpoint' 
          --results './workdir/results' 
          --input './testset_dir/imagesTs' 
          --output './perdiction_output' 
          --plans_identifier 'nnUNetPlans_pretrained_nnUNetData_plans_v2.1' 
          --store_probability_maps 
          'task_name'
```

## 📖 Citation
The method was developed at the [CIMORe](https://www.ntnu.edu/isb/mr-cancer) - Cancer Imaging and Multi-Omics Research Group at the Norwegian University of Science and Technology (NTNU) in Trondheim, Norway. For detailed information about this method, please read our [SPIE medical imaging conference paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12927/129271Q/Federated-learning-for-prostate-cancer-detection-in-biparametric-MRI/10.1117/12.2688568.full) or the [Radiology Artificial Intelligence manuscript](https://www.to_be_add.com). If you use this work, please cite:

```bibtex
@article{moradi2025optimizing,
  title={Optimizing Federated Learning Configurations for MRI Prostate Segmentation and Cancer Detection: A Simulation Study},
  author={Moradi, Ashkan and Zerka, Fadila and Bosma, Joeran Sander and Sunoqrot, Mohammed RS and Abrahamsen, Bendik S and Yakar, Derya and Geerdink, Jeroen and Huisman, Henkjan and Bathen, Tone Frost and Elschot, Mattijs},
  journal={Radiology: Artificial Intelligence},
  pages={e240485},
  year={2025},
  publisher={Radiological Society of North America}
}
```
and
```bibtex
@inproceedings{moradi2024federated,
  title={Federated learning for prostate cancer detection in biparametric MRI: optimization of rounds, epochs, and aggregation strategy},
  author={Moradi, Ashkan and Zerka, Fadila and Bosma, Joeran Sander and Yakar, Derya and Geerdink, Jeroen and Huisman, Henkjan and Bathen, Tone Frost and Elschot, Mattijs},
  booktitle={SPIE Medical Imaging 2024: Computer-Aided Diagnosis},
  volume={12927},
  pages={412--421},
  year={2024}}
``` 


## Acknowledgements
We acknowledge the authors of the publicly available datasets used in this study, whose contributions have enabled valuable research. Additionally, we extend our gratitude to the developers of [Flower](https://flower.ai/), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), and the [PI-CAI](https://pi-cai.grand-challenge.org/) Grand Challenge for making their important contributions publicly accessible (a big thank you to the developers of [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep), [`picai_baseline`](https://github.com/DIAGNijmegen/picai_baseline), and [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval) as these repositories greatly helped us in providing this codebase).
