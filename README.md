# Enhancing Compressed Audio Using Neural Post-Processing

## Overview
 

## Installation

### 1. Install Python 3.13+


### 2. Create environment


#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Windows (using command line)
```bash
python -m venv venv
venv\Scripts\activate
```


### 3. Clone repository and install requirements
```bash
git clone https://github.com/vuminhkh0a/Neural-Post-Processing.git
cd Neural-Post-Processing
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Obtaining datasets (optional)
To obtain the datasets, please read the [Datasets](#datasets) section

## Datasets

**Datasets description**
- Original source
    - [IRMAS dataset](https://zenodo.org/records/1290750#.WzCwSRyxXMU)  
    - [LibriSpeech dataset](https://www.openslr.org/60/)

- For IRMAS dataset, the validation and test dataset are obtained from `IRMAS-TestingData-Part1.zip` and the train dataset are obtained from `IRMAS-TestingData-Part2.zip`

- For LibriSpeech dataset, the train, validation, and test datasets are obtained from `train-clean-100.tar.gz`, `dev-clean.tar.gz`, and `test-clean.tar.gz`, respectively 

**Datasets installation**

- These datasets are not included in this repository due to the `data/` folder's size and some constraints of my local server (no permission to install Git LFS to push that folder since I am not the admin). Please download them from the zip file of the following link: https://drive.google.com/drive/folders/1GB4fvLmA41TZ-W7qMPD9Q0evgFkD-sEI?usp=sharing
, then extract the archive to obtain the `data/` folder and move it to the `Neural-Post-Processing/` folder.

- The `data/` folder tree should be:
```bash
data
    IRMAS
        val
        train
        test
    LibriSpeech
        val
        train
        test
```

## Training and evaluating
