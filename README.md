# Enhancing Compressed Audio Using Neural Post-Processing

## Overview
This is the project for the course ET4262E (Compression and Coding - HUST). We develop a U-Net based model to enhance the compressed audio.  


## Installation

### 1. Install Python 3.12+


### 2. Create environment


#### Linux / macOS
```bash
python3 -m venv npp_env
source npp_env/bin/activate
```
#### Windows (using command line)
```bash
python3 -m venv npp_env
npp_env\Scripts\activate
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
    - [Mini LibriSpeech dataset](https://www.openslr.org/31/)

- For IRMAS dataset, the validation and test dataset are obtained from `IRMAS-TestingData-Part1.zip`, and the train dataset are obtained from `IRMAS-TestingData-Part2.zip`

- For Mini LibriSpeech dataset, the validation and the test dataset are obtained from `dev-clean-2.tar.gz`, and the train dataset is obtained from `train-clean-5.tar.gz`, 

- The degraded audio files are obtained using ADPCM with `num_bits (bit rate) =2`. See [generate.py](generate.py)

**Datasets installation**

- These datasets are not included in this repository due to the `data/` folder's size and some constraints of my sponsor's local server (no permission to install Git LFS to push that folder since I am not the admin). So please download them from the zip file of the following link instead: [Dataset link](https://drive.google.com/drive/folders/1GB4fvLmA41TZ-W7qMPD9Q0evgFkD-sEI?usp=sharing) (~7Gb)
, then extract the archive to obtain the `data/` folder and make sure the `data/` folder is inside the `Neural-Post-Processing/` folder.

- The `data/` folder tree should be:
```bash
data/
├── IRMAS/
│   ├── degraded_wav/
│   │   ├── val/
│   │   ├── train/
│   │   └── test/
│   └── wav/
│       ├── val/
│       ├── train/
│       └── test/
│
├── LibriSpeech/
│   ├── degraded_wav/
        ├── val/
│       ├── train/
│       └── test/
│   └── wav/
        ├── val/
│       ├── train/
│       └── test/
```

## Training and evaluating
**Training script**

```bash
python3 main.py \
  --model_name <Type of model> \
  --dataset_name <Type of dataset> \
  --batch_size <Mini-batch size> \
  --num_workers <Number of DataLoader workers> \
  --pin_memory <true/false> \
  --device cuda \
  --epochs <Number of epochs> \
  --lr <Learning rate>
  --mode <train/eval>
```

For example:
```bash
python3 main.py \
  --model_name Unet \
  --dataset_name IRMAS \
  --batch_size 2 \
  --num_workers 4 \
  --pin_memory True \
  --device cuda:0 \
  --epochs 30 \
  --lr 1e-4 \
  --mode train
```

**Options**
- model_name: `AutoEncoder`, `Unet`, `UnetPlusPlus`, `UnetAttention`, `Proposed`
- dataset_name: `IRMAS`, `LibriSpeech`
- mode: `train`, `eval`

### 5. Live demo
Please download the [model weights](link) before using this feature. Extract the zip file of the link to acquire the `checkpoints/` folder and make sure the `checkpoints/` folder is inside the `Neural-Post-Processing/` folder.
 

Please check the [live demo video](https://www.youtube.com/watch?v=LZ4XNYuigeM).

To run the live demo
```bash
python3 demo_app.py
```
Navigate the program's given link to run demo at your local device.