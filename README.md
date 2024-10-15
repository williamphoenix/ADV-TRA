# United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories (ADV-TRA)
The code for  NeurIPS'24 paper "United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories"

## About The Project
**ADV-TRA** is an intellectual property (IP) protection framework for DNN models based on mdoel fingerprinting.
It generates several adversarial trajectories as fingerprints to verify the ownership of a suspect model.
<br>

## Getting Started
### File Structure 
```
ADV-TRA-master
├── utils
│   ├── data_process.py
│   ├── utils.py
│   ├── models.py
│   └── adv_gen.py
└── main.py
```
There are several parts of the code:

- `data_process.py`: This file mainly contains the preprocessing of the raw dataset, and allocation of the dataset.
- `utils.py`: This file contains the training strategy for the source model. 
- `models.py`: This file contains the internal setting including the architecture of the source models.
- `adv_gen.py`: This file contains the funcations of the whole fingerprinting scheme, including generating the advsarial trajectories as well as verifying the  suspect model with the trajectories.
- `main.py`: The main function of **ADV-TRA**. 
<br>

### Requirements

* python 3.8.15 
* [pytorch](https://pytorch.org/get-started/locally/) 1.12.0 & torchvision 0.13.0 
* CUDA 11.0 and above are recommended (this is for GPU users)
* numpy 1.23.4
* advertorch 0.2.3
* scikit-learn 0.20.0

Before running the project, make sure you have set up the right environment and installed the above required packages.
<br>

### Hyper-parameters 
The settings of **ADV-TRA** are determined in the parameter **args** in **main.py**. Here, we mainly introduce the important hyper-parameters.
- device: which device to use (CPU or GPU). Default: cuda:0.
- dataset: which dataset to evaluate (cifar10, cifar100, or ImageNet). Default: "cifar10".
- num_train: the number of training data for the source model. Default: 50000.
- num_attack: the number of data for lauching removal attacks. Default: 5000.
- initial_lr: initial learning rate for the optimizer. Default: 0.1.
- epochs: epochs of source model training. Default: 200.
- num_trajectories: the number of trajectories, i.e., fingerprints. Default: 100.
- length: the length of bilateral trajectories. Default: 4.
- factor_lc: length control factor to adjust the step size of each step. Default: 0.9.
- factor_re: reduction factor. Default: 0.95.
- threshold: threshold for fingerprint determination. Default: 0.5.
- tra_classes: the number of classes traversed by the trajectory. Default: 10.
- suspect_path: the path of the suspect model. Default: "./suspect_models/model.pth".
<br>

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete fingerprinting process, which is in `main.py`.

```python
def main(args):
    # Data split
    allocate_data(args)
    
    # Train the source model
    model = build_model(args)
    model = train_model(model, args)
    
    # Generate fingerprints
    generate_trajectory(args)
    
    # Verify
    verify_trajectory(args)
```

You can also run main.py using the cmd command.

```python
$ python main.py --dataset "cifar10" --num_trajectories 100
```

<br>

## Note
- The ImageNet dataset can be downloaded from https://image-net.org/.
- A GPU is not required, but we recommend using one to increase the running speed. 

<br>


## Cite Our Paper
```
@inproceedings{,
  title={United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories}, 
  author={Xu, Tianlong and Wang, Chen and Liu, Wei and Yang, Yang and Liu, Gaoyang},
  year={2024},
  booktitle={Proceedings of NeurIPS}
}
```
