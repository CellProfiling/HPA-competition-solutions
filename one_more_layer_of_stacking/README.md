# One More Layer Of Stacking

## Model description

14 CNN models ensembled via LightGBM stacking, optimized with Wadam, using focal and LSEP loss.

## Model source

The original model source can be found [here](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/one_more_layer_of_stacking).

## Model usage

- Run `external_data.py` to download and parse additional data from HPA site.
- Run `convert_to_rgb.py` to fill `input/train_png` with RGB images and `input/train_rgby` with RGBY images.
- Run `pipeline.py` to train and predict with all networks and with stack after that (final .csv file will be at `submits/submission.csv`).
