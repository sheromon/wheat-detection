# Description

The cmd-wheat-detection Python package includes code to train and evaluate a basic model for the [Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection/overview) problem. This deep learning model uses [pytorch](https://pytorch.org/) with [pytorch lightning](https://www.pytorchlightning.ai/) to run [torchvision](https://pytorch.org/vision/stable/index.html) object detection models.

# Usage

Set up virtual environment
```
python3 -m venv venv && pip install --upgrade pip && pip install -e .
```

Activate virtual environment
```
source venv/bin/activate
```

Run training with the default config.ini file
```
python wheat/scripts/train.py
```

## Configuration

Configuration is managed using [configobj](https://configobj.readthedocs.io/en/latest/configobj.html). The full set of configurable parameters with type information is available in [wheat/config/configspec.ini]([wheat/config/configspec.ini]). Overrides to default values can be specified in a custom `config.ini` file.

### Example custom config file
Indentation is not required but is included because I find it helps me see which parameters are in which section or sub-section.
```
['train']
    batch_size = 2
    [['optimizer']]
        initial_lr = 0.0001
```

Run training with a custom configuration
```
python wheat/scripts/train.py --config-path my_config.ini
```

# TODO
* Add data augmentation
* Add learning rate schedule
* Verify that GPU training works
* Get Python entrypoints working
