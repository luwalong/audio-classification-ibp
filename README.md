# audio-classification-ibp
Tensorflow implementation of audio classification, trainable both with vanilla method and with [IBP(Interval Bound Propagation)](https://github.com/deepmind/interval-bound-propagation).

## Overview
*This version is not the complete version.*  
The goal of this project is to demonstrate the vulnerability of a simple neural audio classification models and the effectiveness of the Interval Bound Propagation(IBP).
IBP is a method of abstract interpretation, which dedicates to prove robustness of a program or a model in a deterministic way.
This method splits the input data with **lower bound** and **upper bound** of applicable perturbation.
Using this new data, IBP aims to trace 'sound' bounds through the various operations.
We can insist a model is **robust** at the end of the propagation by checking the result from the worst possible logits, which means the lowest possible value of the correct label's logit is still higher than the other logits' highest possible values.


## Setup
After you clone the repository, you may
1. Prepare the data. Current version of the project supports [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset), so clone the repo on the working directory.
1. Load the virtual environments. Following is the list of the critical packages:
    * Tensorflow = 1.13
    * numpy = 1.14.2 nomkl
    * scipy = 1.1.0 nomkl

## Run

```python . --help``` for listing the command line arguments

### Training

```
python training.py --model_name=<model_name> --use_ibp=<True/False>
```

If you set up ```use_ibp``` as ```True```, then the model will be trained with the combined method of vanilla and IBP.
The resulting model will have higher provability for the data compared to the model trained with ```---use_ibp=False```.

### Testing

```
python testing.py --model_name=<model_name> --use_ibp=<True/False> --test_eps=<negative float>
```

Run the classification through all the test set.
if you set ```--use_ibp=True```, you also can get the provability regarding given ```test_eps```.


## Notes

TODO:
    * Complete comments
    * Carlini attack for demonstrating adversarial inputs
    * Experiment results on README
    * Improved perturbation method
