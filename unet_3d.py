import keras.backend as K
from keras.engine import Input, Model
import keras
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D
import metrics as m
from keras.layers.core import Lambda
import numpy as np
from keras.utils import multi_gpu_model

def up_and_concate_3d(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[4]
    out_channel = in_channel // 2
    up = Conv3DTranspose(out_channel, [2, 2, 2], strides=[2, 2, 2], padding='valid')(down_layer)
    print("--------------")
    print(str(up.get_shape()))

    print(str(layer.get_shape()))
    print("--------------")
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=4))

    concate = my_concat([up, layer])
    # must use lambda
    # concate=K.concatenate([up, layer], 3)
    return concate


def attention_block_3d(x, g, inter_channel):
    '''

    :param x: x input from down_sampling same layer output x(?,x_height,x_width,x_depth,x_channel)
    :param g: gate input from up_sampling layer last output g(?,g_height,g_width,g_depth,g_channel)
    g_height,g_width,g_depth=x_height/2,x_width/2,x_depth/2
    :return:
    '''
    # theta_x(?,g_height,g_width,g_depth,inter_channel)
    theta_x = Conv3D(inter_channel, [2, 2, 2], strides=[2, 2, 2])(x)

    # phi_g(?,g_height,g_width,g_depth,inter_channel)
    phi_g = Conv3D(inter_channel, [1, 1, 1], strides=[1, 1, 1])(g)

    # f(?,g_height,g_width,g_depth,inter_channel)
    f = Activation('relu')(keras.layers.add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,g_depth,1)
    psi_f = Conv3D(1, [1, 1, 1], strides=[1, 1, 1])(f)

    # sigm_psi_f(?,g_height,g_width,g_depth)
    sigm_psi_f = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width,x_depth)
    rate = UpSampling3D(size=[2, 2, 2])(sigm_psi_f)

    # att_x(?,x_height,x_width,x_depth,x_channel)
    att_x = keras.layers.multiply([x, rate])

    return att_x


def unet_model_3d(input_shape, n_labels, batch_normalization=False, initial_learning_rate=0.00001,
                  metrics=m.dice_coef):
    """
    input_shape:without batch_size,(img_height,img_width,img_depth)
    metrics:
    """

    inputs = Input(input_shape)

    down_layer = []

    layer = inputs

    # down_layer_1
    layer = res_block_v2_3d(layer, 32, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')(layer)

    print(str(layer.get_shape()))

    # down_layer_2
    layer = res_block_v2_3d(layer, 64, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')(layer)

    print(str(layer.get_shape()))

    # down_layer_3
    layer = res_block_v2_3d(layer, 128, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')(layer)

    print(str(layer.get_shape()))

    # down_layer_4
    layer = res_block_v2_3d(layer, 256, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')(layer)

    print(str(layer.get_shape()))

    # bottle_layer
    layer = res_block_v2_3d(layer, 512, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_4
    layer = up_and_concate_3d(layer, down_layer[3])
    layer = res_block_v2_3d(layer, 256, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_3
    layer = up_and_concate_3d(layer, down_layer[2])
    layer = res_block_v2_3d(layer, 128, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_2
    layer = up_and_concate_3d(layer, down_layer[1])
    layer = res_block_v2_3d(layer, 64, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_1
    layer = up_and_concate_3d(layer, down_layer[0])
    layer = res_block_v2_3d(layer, 32, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # score_layer
    layer = Conv3D(n_labels, [1, 1, 1], strides=[1, 1, 1])(layer)
    print(str(layer.get_shape()))

    # softmax
    layer = Activation('softmax')(layer)
    print(str(layer.get_shape()))

    outputs = layer

    model = Model(inputs=inputs, outputs=outputs)

    metrics = [metrics]

    model=multi_gpu_model(model, gpus=2)
    model.summary()
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='categorical_crossentropy', metrics=metrics)

    return model


def res_block_v2_3d(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                    padding='same'):

    input_n_filters = input_layer.get_shape().as_list()[4]
    #print(str(input_layer.get_shape()))
    #print(out_n_filters)
    #print(input_n_filters)
    layer = input_layer

    for i in range(2):
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv3D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)


    if out_n_filters != input_n_filters:
        skip_layer = Conv3D(out_n_filters, [1, 1, 1], strides=stride, padding=padding)(input_layer)
    else:
        skip_layer = input_layer

    out_layer = keras.layers.add([layer, skip_layer])


    return out_layer
