from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Permute
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import DepthwiseConv2D
from keras.layers import SpatialDropout2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.constraints import max_norm
from keras.regularizers import l1_l2


# TODO: define optimizer in models, so you can specify learning rate
# TODO: define pool_size, strides and other parameters as experiment
#  properties, tunable from the user in the main script; then passing the
#  entire experiment to the model function constructor or each useful
#  parameter individually


# %% IMPORT MODEL (add here a new if clauses for each new model inserted)
def import_model(dl_experiment):
    if dl_experiment.model_name == 'DeepConvNet':
        fun = DeepConvNet
    elif dl_experiment.model_name == 'DeepConvNet_500Hz':
        fun = DeepConvNet_500Hz
    elif dl_experiment.model_name == 'DeepConvNet_Davide':
        fun = DeepConvNet_Davide
    elif dl_experiment.model_name == 'ShallowConvNet':
        fun = ShallowConvNet
    elif dl_experiment.model_name == 'EEGNet':
        fun = EEGNet
    elif dl_experiment.model_name == 'EEGNet_SSVEP':
        fun = EEGNet_SSVEP
    elif dl_experiment.model_name == 'EEGNet_old':
        fun = EEGNet_old
    elif dl_experiment.model_name == 'DeepConvNet_500':
        fun = DeepConvNet_500Hz
    else:
        raise ValueError('specified model_name is not a valid one;\n' +
                         'consider to add it in hgdecode>models.py')
    # creating the model with the chosen architecture
    model = fun(
        n_classes=dl_experiment.n_classes,
        n_channels=dl_experiment.n_channels,
        n_samples=dl_experiment.crop_sample_size,
        dropout_rate=dl_experiment.dropout_rate
    )
    return model


# %% DEEP CONV NET
def DeepConvNet(n_classes=4,
                n_channels=64,
                n_samples=256,
                dropout_rate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.
    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    Note that this implementation has not been verified by the original
    authors.
    """

    # start the model
    input_main = Input((1, n_channels, n_samples))
    block1 = Conv2D(25, (1, 10),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (n_channels, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = Conv2D(50, (1, 10),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropout_rate)(block2)

    block3 = Conv2D(100, (1, 10),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropout_rate)(block3)

    block4 = Conv2D(200, (1, 10),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropout_rate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(n_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# %% DEEP CONV NET 500 Hz
def DeepConvNet_500Hz(n_classes=4,
                      n_channels=64,
                      n_samples=256,
                      dropout_rate=0.5):
    """
    # TODO: description for this model
    """
    # input
    input_main = Input((1, n_channels, n_samples))

    # block1
    block1 = Conv2D(25, (1, 20),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(input_main)
    block1 = Conv2D(25, (n_channels, 1),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropout_rate)(block1)

    # block2
    block2 = Conv2D(50, (1, 20),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropout_rate)(block2)

    # block3
    block3 = Conv2D(100, (1, 20),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropout_rate)(block3)

    # block4
    block4 = Conv2D(200, (1, 20),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block3)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropout_rate)(block4)

    # flatten
    flatten = Flatten()(block4)

    # another dense one
    # dense = Dense(128, bias_initializer='truncated_normal',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(0.001),
    #              kernel_constraint=max_norm(0.5))(flatten)
    # dense = Activation('elu')(dense)
    # dense = Dropout(dropout_rate)(dense)

    # dense
    dense = Dense(n_classes,
                  # bias_initializer='truncated_normal',
                  # kernel_initializer='truncated_normal',
                  kernel_constraint=max_norm(0.5)
                  )(flatten)
    softmax = Activation('softmax')(dense)

    # returning the model
    return Model(inputs=input_main, outputs=softmax)


# %% DEEP CONV NET DAVIDE
def DeepConvNet_Davide(n_classes=4,
                       n_channels=64,
                       n_samples=256,
                       dropout_rate=0.5):
    """
    TODO: a description for this model
    :param n_classes:
    :param n_channels:
    :param n_samples:
    :param dropout_rate:
    :return:
    """
    # start the model
    input_main = Input((1, n_channels, n_samples))
    block1 = Conv2D(25, (1, 10),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(input_main)
    block1 = Conv2D(25, (n_channels, 1),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = Conv2D(50, (1, 10),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
    block2 = Dropout(dropout_rate)(block2)

    block3 = Conv2D(100, (1, 10),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
    block3 = Dropout(dropout_rate)(block3)

    block4 = Conv2D(200, (1, 10),
                    # bias_initializer='truncated_normal',
                    # kernel_initializer='he_normal',
                    # kernel_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2))
                    )(block3)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
    block4 = Dropout(dropout_rate)(block4)

    flatten = Flatten()(block4)
    # dense = Dense(128, bias_initializer='truncated_normal',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(0.001),
    #              kernel_constraint=max_norm(0.5))(flatten)
    # dense = Activation('elu')(dense)
    # dense = Dropout(dropout_rate)(dense)
    dense = Dense(n_classes,
                  # bias_initializer='truncated_normal',
                  # kernel_initializer='truncated_normal',
                  kernel_constraint=max_norm(0.5)
                  )(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# %% SHALLOW CONV NET
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(n_classes,
                   n_channels=64,
                   n_samples=128,
                   dropout_rate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((1, n_channels, n_samples))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, n_channels, n_samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (n_channels, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropout_rate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(n_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# %% EEG NET
def EEGNet(n_classes,
           n_channels=64,
           n_samples=128,
           dropout_rate=0.25,
           kernel_length=64,
           F1=4,
           D=2,
           F2=8,
           norm_rate=0.25,
           dropout_type='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-4,2 model as discussed
    in the paper. This model should do pretty well in general, although as the
    paper discussed the EEGNet-8,2 (with 8 temporal kernels and 2 spatial
    filters per temporal kernel) can do slightly better on the SMR dataset.
    Other variations that we found to work well are EEGNet-4,1 and EEGNet-8,1.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of
    this parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D
    for overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      n_classes     : int, number of classes to classify
      n_channels    : number of channels
      n_samples     : number of time points in the EEG data
      dropout_rate  : dropout fraction
      kernel_length : length of temporal convolution in first layer. We found
                    that setting this to be half the sampling rate worked
                    well in practice. For the SMR dataset in particular
                    since the data was high-passed at 4Hz we used a kernel
                    length of 32.
      F1, F2        : number of temporal filters (F1) and number of pointwise
                    filters (F2) to learn. Default: F1 = 4, F2 = F1 * D.
      D             : number of spatial filters to learn within each temporal
                    convolution. Default: D = 2
      dropout_type  : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropout_type == 'SpatialDropout2D':
        dropout_type = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = Dropout
    else:
        raise ValueError('dropout_type must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, n_channels, n_samples))

    ##################################################################
    block1 = Conv2D(F1, (1, kernel_length), padding='same',
                    input_shape=(1, n_channels, n_samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((n_channels, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_type(dropout_rate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_type(dropout_rate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(n_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# %% EEG NET SSVEP
def EEGNet_SSVEP(n_classes=12,
                 n_channels=8,
                 n_samples=256,
                 dropout_rate=0.5,
                 kernel_length=256,
                 F1=96,
                 D=1,
                 F2=96,
                 dropout_type='Dropout'):
    """ SSVEP Variant of EEGNet, as used in [1].
    Inputs:

      n_classes     : int, number of classes to classify
      n_channels    : number of channels
      n_samples     : number of time points in the EEG data
      dropout_rate  : dropout fraction
      kernel_length : length of temporal convolution in first layer
      F1, F2        : number of temporal filters (F1) and number of pointwise
                    filters (F2) to learn.
      D             : number of spatial filters to learn within each temporal
                    convolution.
      dropout_type  : Either SpatialDropout2D or Dropout, passed as a string.


    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6).
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8
    """

    if dropout_type == 'SpatialDropout2D':
        dropout_type = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = Dropout
    else:
        raise ValueError('dropout_type must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, n_channels, n_samples))

    ##################################################################
    block1 = Conv2D(F1, (1, kernel_length), padding='same',
                    input_shape=(1, n_channels, n_samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((n_channels, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_type(dropout_rate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_type(dropout_rate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(n_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# %% EEG NET OLD
def EEGNet_old(n_classes,
               n_channels=64,
               n_samples=128,
               regRate=0.0001,
               dropout_rate=0.25,
               kernels=None,
               strides=(2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)
    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2

    with a few modifications: we use striding instead of max-pooling as this
    helped slightly in classification performance while also providing a
    computational speed-up.

    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.

    Inputs:

        n_classes    : total number of final categories
        n_channels   : number of EEG channels
        n_samples    : number of EEG time points
        regRate      : regularization rate for L1 and L2 regularizations
        dropout_rate : dropout fraction
        kernels      : the 2nd and 3rd layer kernel dimensions (default is
                       the [2, 32] x [8, 4] configuration)
        strides      : the stride size (note that this replaces the max-pool
                       used in the original paper)

    """
    # fixing PEP8 mutable input issue
    if kernels is None:
        kernels = [(2, 32), (8, 4)]

    # start the model
    input_main = Input((1, n_channels, n_samples))
    layer1 = Conv2D(16, (n_channels, 1),
                    input_shape=(1, n_channels, n_samples),
                    kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(
        input_main)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = Dropout(dropout_rate)(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(4, kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(permute1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = Dropout(dropout_rate)(layer2)

    layer3 = Conv2D(4, kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = Dropout(dropout_rate)(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(n_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
