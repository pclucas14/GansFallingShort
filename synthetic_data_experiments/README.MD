## Synthetic Task
All files specific to the synthetic task are in this folder


### Reproducing Figure 3

Running
```
python oracle_eval.py
```
will launch the best run for each method (selected via cross validation). This includes

- MLE
- RL-GAN
- CoT

The tensorboard logs will be saved in `reproduce_synthetic_results`.

Results for LeakGAN and SeqGAN were obtained by running the respective original repositories. The results (as well as all the ones needed to plot figure 3) are stored in `../csv/oracle`

The code to plot the actual graph can be found in `notebooks/oracle_graphs.ipynb`

If you want to plot the results of runs in  `reproduce_synthetic_results`, you will need to open TB, save the `.csv` and put them in `../csv/oracle` or modify the plotting code to fetch the new csvs.

