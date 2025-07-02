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




## Installation:
Clone the entire repository to your local system, and then install the environment using the `.yml` file: `conda env create -f environment.yml`. Next, install the pipelines developed by the PI-CAI Grand Challenge organizers for data conversion (DICOM Archive → MHA Archive, MHA Archive → nnU-Net Raw Data Archive). Details for data conversion and evaluation, following the PI-CAI Grand Challenge guidelines, are provided in [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) and [`picai_eval`](https://github.com/DIAGNijmegen/picai_eval), respectively. Then, the modified version of the nnUNet codebase ([version 1.7.0](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)) and Flower must be installed:

```bash
cd nnUNet  
pip install -e .
pip install flwr==1.3.0
```







The method was developed at the [CIMORe](https://www.ntnu.edu/isb/mr-cancer) - Cancer Imaging and Multi-Omics Research Group at the Norwegian University of Science and Technology (NTNU) in Trondheim, Norway. For detailed information about this method, please read our [SPIE medical imaging conference paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12927/129271Q/Federated-learning-for-prostate-cancer-detection-in-biparametric-MRI/10.1117/12.2688568.full) or the [Radiology Artificial Intelligence manuscript](https://www.to_be_add.com). 


Complete details about the implementation and the required software and packages will be made publicly available upon publication of the manuscript (stay tuned...)

