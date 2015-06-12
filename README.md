# mnist-nengo
Basic network(s) for classifying MNIST digits in Nengo

## Basic usage
To train a new network, run

    python train.py

This will train a network and save it to a `.npz` file starting with `params`.
The optional `--gpu` flag runs on the GPU, and the save file can be specified.

To run a trained network in spiking neurons, do

    python run.py params_file.npz output_file.npz

where `params_file.npz` is the name of your trained network params file
(or one of the pretrained files `lif-126-error.npz` or `lif-111-error.npz`),
and `output_file.npz` is an optional location to save the output.

If you do choose to save your output, you can view it again with

    python view.py output_file.npz

You can also get some information about static trained networks with

    python view.py params_file.npz

You can also run any of the above scripts with the `--help` argument to get
a full list of arguments.

## Requirements
This project requires Nengo, and additionally Theano and Scipy if you
want to train your own networks. Both should be installable from `pip`,
but using Theano on the GPU requires CUDA to also be installed
([details](http://deeplearning.net/software/theano/tutorial/using_gpu.html)).
