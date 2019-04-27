import numpy as np
import os
import glob
import SimpleITK as sitk
import random
import util


def sitk_read(img_path):
    nda = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(nda) #(155,240,240)
    zero = np.zeros([5, 240, 240])
    nda = np.concatenate([zero, nda], axis=0) #(160,240,240)
    nda = nda.transpose(1, 2, 0) #(240,240,160)
    return nda


def sitk_read_row(img_path):
    nda = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(nda) #(155,240,240)
    return nda


def make_one_hot_3d(x, n):
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1
    return one_hot


class brast_reader:
    def __init__(self, train_batch_size, val_batch_size, test_batch_size, type='HGG'):
        self.train_root_path = "/root/userfolder/PY/data/BRAST2015/BRATS2015_Training/"
        self.type = type


        self.train_name_list = self.load_file_name_list(self.train_root_path + "train_name_list.txt")
        self.val_name_list = self.load_file_name_list(self.train_root_path + "val_name_list.txt")
        self.test_name_list = self.load_file_name_list(self.train_root_path + "test_name_list.txt")

        self.n_train_file = len(self.train_name_list)
        self.n_val_file = len(self.val_name_list)
        self.n_test_file = len(self.test_name_list)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.n_train_steps_per_epoch = self.n_train_file // self.train_batch_size
        self.n_val_steps_per_epoch = self.n_val_file // self.val_batch_size

        self.img_height = 240
        self.img_width = 240
        self.img_depth = 160
        self.n_labels = 5

        self.train_batch_index = 0
        self.val_batch_index = 0


    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        return file_name_list

    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.train_root_path + self.type + "/")
        random.shuffle(data_name_list)
        length = len(data_name_list)
        n_train_file = int(length / 10 * 8)
        n_val_file = int(length / 10 * 1)
        train_name_list = data_name_list[0:n_train_file]
        val_name_list = data_name_list[n_train_file:(n_train_file + n_val_file)]
        test_name_list = data_name_list[(n_train_file + n_val_file):len(data_name_list)]
        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")
        self.write_name_list(test_name_list, "test_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.train_root_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(name_list[i] + "\n")
        f.close()

    def next_train_batch_2d(self):
        if self.train_batch_index >= self.n_train_file:
            self.train_batch_index = 0

        data_path = self.train_root_path + self.type + '/' + self.train_name_list[self.train_batch_index]

        # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
        t1, ot = self.get_np_data_2d(data_path)
        train_imgs=t1[:,:,:,np.newaxis] #(155,240,240,1)
        train_labels=make_one_hot_3d(ot,self.n_labels) #(155,240,240,5)

        self.train_batch_index+=1
        return train_imgs,train_labels

    def next_val_batch_2d(self):
        if self.val_batch_index >= self.n_val_file:
            self.val_batch_index = 0

        data_path = self.train_root_path + self.type + '/' + self.val_name_list[self.val_batch_index]


        # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
        t1, ot = self.get_np_data_2d(data_path)

        val_imgs=t1[:,:,:,np.newaxis] #(155,240,240,1)
        val_labels=make_one_hot_3d(ot,self.n_labels)
        self.val_batch_index += 1

        return val_imgs, val_labels

    def next_train_batch_3d_sub(self,num_sub,sub_size):
        train_imgs = np.zeros((num_sub, sub_size[0], sub_size[1], sub_size[2], 1))
        train_labels = np.zeros([num_sub, sub_size[0], sub_size[1], sub_size[2], self.n_labels])
        if self.train_batch_index >= self.n_train_steps_per_epoch:
            self.train_batch_index = 0
        for i in range(self.train_batch_size):
            data_path = self.train_root_path + self.type + '/' + self.train_name_list[
                self.train_batch_size * self.train_batch_index + i]

            for j in range(num_sub):
                # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
                t1, ot = self.get_np_data_3d(data_path)
                t1,ot=util.random_crop_3d(t1,ot,sub_size)
                # flair=flair[:,:,:,np.newaxis]
                t1 = t1[:, :, :, np.newaxis]
                # t1c = t1c[:, :, :, np.newaxis]
                # t2 = t2[:, :, :, np.newaxis]
                train_imgs[j] = t1
                one_hot = make_one_hot_3d(ot, self.n_labels)
                train_labels[j] = one_hot

        self.train_batch_index += 1
        #print(train_imgs.shape)
        return train_imgs, train_labels

    def next_val_batch_3d_sub(self, num_sub, sub_size):
        val_imgs = np.zeros((num_sub, sub_size[0], sub_size[1], sub_size[2], 1))
        val_labels = np.zeros([num_sub, sub_size[0], sub_size[1], sub_size[2], self.n_labels])
        if self.val_batch_index >= self.n_val_steps_per_epoch:
            self.val_batch_index = 0
        for i in range(self.val_batch_size):
            data_path = self.train_root_path + self.type + '/' + self.val_name_list[
                self.val_batch_size * self.val_batch_index + i]

            for j in range(num_sub):
                # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
                t1, ot = self.get_np_data_3d(data_path)
                t1, ot = util.random_crop_3d(t1, ot, sub_size)
                # flair=flair[:,:,:,np.newaxis]
                t1 = t1[:, :, :, np.newaxis]
                # t1c = t1c[:, :, :, np.newaxis]
                # t2 = t2[:, :, :, np.newaxis]
                val_imgs[j] = t1
                one_hot = make_one_hot_3d(ot, self.n_labels)
                val_labels[j] = one_hot

        self.val_batch_index += 1
        # print(train_imgs.shape)
        return val_imgs, val_labels

    def next_train_batch_3d(self):
        train_imgs = np.zeros((self.train_batch_size, self.img_height, self.img_width, self.img_depth, 1))
        train_labels = np.zeros([self.train_batch_size, self.img_height, self.img_width, self.img_depth, self.n_labels])
        if self.train_batch_index >= self.n_train_steps_per_epoch:
            self.train_batch_index = 0
        for i in range(self.train_batch_size):
            data_path = self.train_root_path + self.type + '/' + self.train_name_list[
                self.train_batch_size * self.train_batch_index + i]

            # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
            t1, ot = self.get_np_data_3d(data_path)
            # flair=flair[:,:,:,np.newaxis]
            t1 = t1[:, :, :, np.newaxis]
            # t1c = t1c[:, :, :, np.newaxis]
            # t2 = t2[:, :, :, np.newaxis]
            train_imgs[i] = t1
            one_hot = make_one_hot_3d(ot, self.n_labels)
            train_labels[i] = one_hot

        self.train_batch_index += 1
        #print(train_imgs.shape)
        return train_imgs, train_labels

    def next_val_batch_3d(self):
        val_imgs = np.zeros((self.train_batch_size, self.img_height, self.img_width, self.img_depth, 1))
        val_labels = np.zeros([self.train_batch_size, self.img_height, self.img_width, self.img_depth, self.n_labels])
        if self.val_batch_index >= self.n_val_steps_per_epoch:
            self.val_batch_index = 0
        for i in range(self.val_batch_size):
            data_path = self.train_root_path + self.type + '/' + self.val_name_list[
                self.val_batch_size * self.val_batch_index + i]

            # flair, t1, t1c, t2, ot=self.get_np_data(data_path)
            t1, ot = self.get_np_data_3d(data_path)
            # flair=flair[:,:,:,np.newaxis]
            t1 = t1[:, :, :, np.newaxis]
            # t1c = t1c[:, :, :, np.newaxis]
            # t2 = t2[:, :, :, np.newaxis]
            val_imgs[i] = t1
            one_hot = make_one_hot_3d(ot, self.n_labels)
            val_labels[i] = one_hot

        self.val_batch_index += 1

        return val_imgs, val_labels

    def get_np_data_3d(self, data_path):

        t1_file_path=''
        ot_file_path=''
        t1=os.path.join(data_path, 'VSD.Brain.XX.O.MR_T1.*/VSD.Brain.XX.O.MR_T1.*.mha')
        ot=os.path.join(data_path, 'VSD.Brain*.XX.*.OT.*/VSD.Brain*.XX.*.OT.*.mha')
        '''
        print("----")
        print(t1)
        print(ot)
        print("----")
        '''
        for i in glob.glob(t1):
            t1_file_path = i
        for i in glob.glob(ot):
            ot_file_path = i

        '''
        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_Flair.*/VSD.Brain.XX.O.MR_Flair.*.mha')):
            flair_file_path=i

        

        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_T1c.*/VSD.Brain.XX.O.MR_T1c.*.mha')):
            t1c_file_path=i

        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_T2.*/VSD.Brain.XX.O.MR_T2.*.mha')):
            t2_file_path=i

        

        flair=sitk_read(flair_file_path)
        
        t1c=sitk_read(t1c_file_path)
        t2=sitk_read(t2_file_path)
        
        '''
        t1 = sitk_read(t1_file_path)
        ot = sitk_read(ot_file_path)
        return t1, ot
        # return flair,t1,t1c,t2,ot

    def get_np_data_2d(self, data_path):
        for i in glob.glob(os.path.join(data_path, 'VSD.Brain.XX.O.MR_T1.*/VSD.Brain.XX.O.MR_T1.*.mha')):
            t1_file_path = i
        for i in glob.glob(os.path.join(data_path, 'VSD.Brain_3more.XX.*/VSD.Brain_3more.XX.*.mha')):
            ot_file_path = i

        '''
        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_Flair.*/VSD.Brain.XX.O.MR_Flair.*.mha')):
            flair_file_path=i



        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_T1c.*/VSD.Brain.XX.O.MR_T1c.*.mha')):
            t1c_file_path=i

        for i in glob.glob(os.path.join(data_path,'VSD.Brain.XX.O.MR_T2.*/VSD.Brain.XX.O.MR_T2.*.mha')):
            t2_file_path=i



        flair=sitk_read(flair_file_path)

        t1c=sitk_read(t1c_file_path)
        t2=sitk_read(t2_file_path)

        '''
        t1 = sitk_read_row(t1_file_path)
        ot = sitk_read_row(ot_file_path)
        return t1, ot
        # return flair,t1,t1c,t2,ot


def main():
    reader = brast_reader(2, 2, 2)
    reader.write_train_val_test_name_list()



if __name__ == '__main__':
    main()
