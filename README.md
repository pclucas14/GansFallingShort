# Language GANs Falling Short

Code for reproducing all results in our paper, which can be found [here](https://arxiv.org/abs/1811.02549)


## (key) Requirements 
- Python 3.x
- Pytorch 0.4.1
- TensorboardX

## Structure
- `common` folder: most of the important code is here, including all models and utilities
- `synthetic_data_experiments` folder: code to run and reproduce all oracle experiments
- `real_data_experiments` folder: code to run results for ImageCoco and News datasets

## Reproducibility
- For synthetic data, simply run `oracle_eval.py` found in the `synthetic_data_experiments` folder. 
- For real data, we uploaded the weights (and corresponding hyperparameters) in `real_data_experiments/trained_models` folder. You can load the model by using the `--load_{gen/disc}_from_file` argument. For example, 
```
python main.py --load_gen_path trained_models/news/word/best_mle
```

## Contact
For any questions / comments / concerns, feel free to open an issue via github, or to send me an email at <br /> `lucas.page-caccia@mail.mcgill.ca`. <br />

We strongly believe in fully reproducible research. To that end, if you find any discrepancy between our code and the paper, please let us know, and we will make sure to address it.  <br />

Happy text modeling :)
