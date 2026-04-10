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

- The degraded audio files are obtained using ADPCM with `num_bits=2`. See [generate.py](generate.py)

**Datasets installation**

- These datasets are not included in this repository due to the `data/` folder's size and some constraints of my sponsor's local server (no permission to install Git LFS to push that folder since I am not the admin). So please download them from the zip file of the following link instead: [Dataset link](https://drive.google.com/drive/folders/1GB4fvLmA41TZ-W7qMPD9Q0evgFkD-sEI?usp=sharing) (~23Gb)
, then extract the archive to obtain the `data/` folder and move it to the `Neural-Post-Processing/` folder.

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
  --input_type <Type of input form> \
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
python main.py \
  --model_name Unet \
  --dataset_name IRMAS \
  --input_type spectrogram \
  --batch_size 2 \
  --num_workers 4 \
  --pin_memory True \
  --device cuda:0 \
  --epochs 30 \
  --lr 1e-4 \
  --mode train
```

**Options**
- model_name: `Unet`, `AutoEncoder`
- dataset_name: `IRMAS`, `LibriSpeech`
- input_type: `spectrogram`, `waveform`
- mode: `train`, `eval`