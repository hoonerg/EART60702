## Link to github Repository:
https://github.com/hoonerg/EART60702

## How to run the code

### Setup
Run the following to install a subset of necessary python packages for our code
```sh
conda env create -f environment.yml
```

### Data
Download data from https://www.dropbox.com/scl/fo/dmabz9pf3167l62612h5b/h?rlkey=ge8u486w7w7vq8vnpr2f1fvag&dl=0
and place it in ./data/

## How to train the models

### Data preprocessing
This will generate preprocessed npy files from raw data. Root directory has to be modified.
```sh
python data_preprocessing.py
```

### Train / Test
This is for training and testing model.
```sh
python train.py
python test.py
```

## Structure

```sh
.
├── data
│   └── ~.nc # Raw data
├── processed
│   └── checkpoints
│       ├── training.npy
│       ├── validation.npy
│       ├── test_1.npy
│       ├── test_2.npy
│       └── test_3.npy
├── model
│   ├── model_checkpoint_*.pth # checkpoint saved here
│   └── loss_plot.png # Loss plot
├── .gitignore
├── README.md
├── environment.yml # Environment
├── data_preprocessing.py  # Data preprocessing (.nc -> .npy)
├── train.py  # training
├── test.py  # test
└── plotting.ipynb # Plotting figures

```