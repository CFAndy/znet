# luna16
LUNA16 Lung Nodule Analysis - NWI-IMC037 Final Project 

Respository containing code for our final project of the computer aided medical diagnosis course, which yielded an entry in the LUNA16 competition. 
Unfortunately this project is ill documented, as it was created very quickly and pragmatically as we hurried to meet deadlines for the university course this was the main project of. Running it will likely require minor edits left and right (mostly to do with filepaths), unless you place the data in the same folder as we did. 

A rough list of requirements:

**Python 2.7** with `theano, lasagne (with CUDNN configured), tqdm, pandas, numpy, scipy, scikit-image, scikit-learn, opencv2`.

## UNET for dense prediction

### Preprocessing
Convert data to 1x1x1mm_512x512 slices. A requirement is also a set of segmentations of the lungs (can be found on the LUNA16 website). Place your data in folder `data/original_lungs`:


Use script `src/data_processing/create_same_spacing_data_NODULE.py`, this may take a long time (up to 24 hours) depending on your machine.

Then, download `imagename_zspacing.csv` from [here](https://gzuidhof.stackstorage.com/s/qsqz9dERe7atYU5) and put it in the data folder.

### Unet training
```
src/deep/unet/caffe_unet_train.sh
```

Parameter config is in params.py

### Unet held out set dense predictions
```
src/deep/caffe_unet_predict.sh
```
When training a model a  folder is created in `/snapshots/unet`. The folder name is the model name here. Manually look up which epoch had the lowest validation set loss and was a checkpoint.

### Unet dense predictions -> candidates:
```
cp -rf ./snapshots/unet_iter_311061.caffemodel_predictions/* ./results
python candidates.py
python candidate_merging.py

a finalizedcandidates_unet_ALL.csv will be generated in data folder
```

## False positive reduction (Wide ResNet)

### Preprocessing:
```
cp ./data/finalizedcandidates_unet_ALL.csv ./csv
. ./src/data_processing/create_wrn_input.sh
```

### Training
. ./src/deep/resnet/caffe_resnet_train.sh

### Predict

```
. ./src/deep/resnet/caffe_resnet_predict.sh
```

The prediction CSV can then be found in the model folder. All you have to do now is combine these. You could use `src/ensembleSubmissions.py` for this, which also features some equalization of predictions of the different models.
