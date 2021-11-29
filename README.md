# Description

The cmd-wheat-detection Python package includes code to train and evaluate a basic model for the [Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection/overview) problem. This deep learning model uses [pytorch](https://pytorch.org/) with [pytorch lightning](https://www.pytorchlightning.ai/) to run [torchvision](https://pytorch.org/vision/stable/index.html) object detection models.

You can see how I use this code in a Kaggle notebook [here](https://www.kaggle.com/sheromon/wheat-faster-r-cnn-with-pytorch-lightning/notebook).

# Usage

Set up virtual environment
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Activate virtual environment
```
source venv/bin/activate
```

## Command-line scripts
Run training as a script using the default configuration
```
train  # using python entry point
```
or
```
python wheat/scripts/train.py
```

As documented in the [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags), PyTorch Lightning CLI flags can be used to configure the training run. For example, to set the number of training epochs for to 10, add `--max_epochs=10` to the train command.

Run evaluation on the validation set with weights from a saved checkpoint
```
evaluate <path-to-checkpoint> # using python entry point
```
or
```
python wheat/scripts/evaluate.py <path-to-checkpoint>
```

Run inference on the test set and save predictions in the Kaggle submission format
```
predict <path-to-checkpoint> # using python entry point
```
or
```
python wheat/scripts/predict.py <path-to-checkpoint>
```

## Jupyter notebook
To install Jupyter notebook support, with the virtual environment active, run `pip install -e .[notebook]`, then start Jupyter.

### Provided notebooks
  * Visualize dataset examples: [notebooks/visualize_dataset.ipynb](notebooks/visualize_dataset.ipynb)
  * Run training: [notebooks/train.ipynb](notebooks/train.ipynb)
  * Run evaluation and analyze results: [notebooks/evaluate.ipynb](notebooks/evaluate.ipynb)

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
train --config-path my_config.ini
```

# TODO

Honestly, I'm not sure when I'll have time to do any of these, but it's nice to have goals, right?

* Get rid of "numpy array is not writeable" error
* Replace config with PyTorch Lightning hyperparameters
* Maybe replace numpy seed with PyTorch Lightning seed feature
* Add stochastic weight averaging
* Add tests?
* Add test-time augmentation?
