import train
import unet_3d
import unet_2d
import brast_reader
import numpy as np
from keras.models import load_model
import metrics as m


def main(argv=None):
    train_3d_sub()
    #predict()


def train_3d_sub():
    sub_size=[64,64,64]
    input_shape=[sub_size[0],sub_size[1],sub_size[2],1]
    n_labels=5
    num_sub=16
    model = unet_3d.unet_model_3d(input_shape, n_labels=n_labels)
    train_batch_size = 1
    val_batch_size = 1
    brast = brast_reader.brast_reader(train_batch_size, val_batch_size, 2)
    n_train_steps_per_epoch = 176 // train_batch_size
    n_val_steps_per_epoch = 22 // val_batch_size

    train.train_model(model, "model_file_3d_sub", train.train_generator_data_3d_sub(brast,sub_size,num_sub),
                      train.val_generator_data_3d_sub(brast,sub_size,num_sub), n_train_steps_per_epoch, n_val_steps_per_epoch,
                      n_epochs=50)

def train_3d():
    input_shape = [240, 240, 160, 1]
    n_labels = 5
    model = unet_3d.unet_model_3d(input_shape, n_labels=n_labels)
    train_batch_size = 2
    val_batch_size = 2
    brast = brast_reader.brast_reader(train_batch_size, val_batch_size, 2)
    n_train_steps_per_epoch = 176 // train_batch_size
    n_val_steps_per_epoch = 22 // val_batch_size

    train.train_model(model, "model_file", train.train_generator_data_3d(brast),
                      train.val_generator_data_3d(brast), n_train_steps_per_epoch, n_val_steps_per_epoch,
                      n_epochs=20)

def train_2d():
    input_shape=[240,240,1]
    n_labels=5
    model=unet_2d.unet_model_2d(input_shape,n_labels)
    train_batch_size=155
    val_batch_size=155
    brast=brast_reader.brast_reader(train_batch_size,val_batch_size,155)
    n_train_steps_per_epoch=176
    n_val_steps_per_epoch=22

    train.train_model(model,"model_file",train.train_generator_data_2d(brast),train.val_generator_data_2d(brast),n_train_steps_per_epoch, n_val_steps_per_epoch)


def predict():
    model=load_model("model_file",custom_objects={'dice_coef':m.dice_coef_test})
    train_batch_size = 2
    val_batch_size = 2
    brast = brast_reader.brast_reader(train_batch_size, val_batch_size, 2)
    for i in range(10):
        x,y=brast.next_train_batch_3d()
        # res=model.predict(x,batch_size=2)
        res=model.evaluate(x, y, batch_size=2)
        print(res)

if __name__ == '__main__':
    main()
