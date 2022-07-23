# Asynchronous Federated Optimization

This is the source code for the paper [Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934)

### Requirements

The following python packages needs to be installed by pip:

1. MXNET (we use Intel CPU cluster, thus mxnet-mkl is preferred)
2. Gluon-CV
3. Numpy
4. Keras (with Tensorflow backend, we use this only for dataset preparation, not for model training)
5. PIL (also for dataset preparation)

The users can simply run the following commond in their own virtualenv:

```bash
pip install --no-cache-dir numpy mxnet-mkl gluoncv keras pillow
```

### Prepare the dataset

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--output DATASET_DIR| the directory where the dataset will be placed|
|--nsplit 100| partition to 100 devices|
|--normalize 1| Normalize the data|

* partition CIFAR-10 dataset:
```bash
python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output DATASET_DIR
```


#### Note that balanced partition is always needed for validation.

### Run the demo

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--dir DATASET_DIR| the directory where the training dataset is placed|
|--valdir VAL_DATASET_DIR| the directory where the validation dataset is placed|
|--batchsize 50| batch size of the workers|
|--epochs 2000| total number of epochs|
|--interval 10| log interval|
|--nsplit 100| training data is partitioned to 100 devices|
|--lr 0.1| learning rate|
|--rho 0.01| regularization weight, different from the \rho in the paper, this is \gamma \times \rho|
|--alpha 0.8| mixing hyperparameter|
|--log | path to the log file|
|--classes 10| number of different classes/labels|
|--iterations 1| number of local epochs in each global epoch|
|--alpha-decay | alpha decay rate|
|--alpha-decay-epoch | epochs where alpha decays|
|--alpha-type | type of adaptive alpha, options are 'none', 'power', 'hinge'|
|--alpha-adaptive | hyperparameter of adaptive alpha (a)|
|--alpha-adaptive2 | hyperparameter of adaptive alpha (b)|
|--max-delay | maximum of global delay|
|--model default | name of the model, "default" means the CNN used in the paper experiments|
|--seed 337 | random seed|

* Train with 100 workers/partitions, on default model:
```bash
python fed_async_paper/train_cifar10_mxnet_fedasync_singlethread_impl.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --rho 0.01 --alpha 0.8 --alpha-decay 0.5 --alpha-decay-epoch 800 --epochs 2000 --max-delay 12 --iterations 1 --seed 336 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
```

More detailed commands/instructions can be found in the demo script *experiment_script_1.sh*

### Notes:

The experiments are executed on the Intel vLab Academic Cluster, some environment variables in the script "experiment_script_1.sh" might be found unnecessary if run in different environments.

