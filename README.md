# Language GANs Falling Short

Code for reproducing all results in our paper, which can be found [here](https://arxiv.org/abs/1811.02549)


## (key) Requirements 
- Python 3.6
- Pytorch 0.4.1
- TensorboardX

## Structure
- `common` folder: most of the important code is here, including all models and utilities
- `synthetic_data_experiments` folder: code to run and reproduce all oracle experiments
- `real_data_experiments` folder: code to run results for ImageCoco and News datasets

## Reproducibility
- For synthetic experiments, we provide the best hyperparameters for every method to reproduce the results. We also provide the [hyperparameter script](https://github.com/pclucas14/GansFallingShort/blob/master/scripts/synthetic_rs.py) that was used to launch the hyperparameter search. More info can be found [here](https://github.com/pclucas14/GansFallingShort/tree/master/synthetic_data_experiments#synthetic-task)

- For real data, we uploaded the weights (and corresponding hyperparameters) in `real_data_experiments/trained_models` folder. We also provie the [hyperparameter script](https://github.com/pclucas14/GansFallingShort/blob/master/real_data_experiments/scripts/news_rs.py) used for this experiment.
We give more detail on reproducing the EMNLPNEWS dataset results
[here](https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments#real-data-experiments)

## Contact
This repository is no longer maintained. Feel free to file an issue if need be, however response may be slow. 
