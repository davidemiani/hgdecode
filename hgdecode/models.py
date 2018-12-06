from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.engine.sequential import Sequential


# %% import_model (add here a new if clauses for each new model inserted)
def import_model(dl_experiment):
    if dl_experiment.model_name == 'MNIST_0':
        model = MNIST_0(dl_experiment)
        return model
    elif dl_experiment.model_name == 'CIFAR10_0':
        model = CIFAR10_0(dl_experiment)
        return model
    elif dl_experiment.model_name == 'SchirmeisterDeepConvNet':
        model = SchirmeisterDeepConvNet(
            dl_experiment.classes,
            dl_experiment.channels,
            dl_experiment.s
        )
    else:
        raise ValueError('Specified model_name is not a valid one.\n' +
                         'You should add it in dmdlmodules>models.py')
