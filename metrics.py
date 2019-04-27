from keras import backend as K


'''
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

'''
def dice_coef_test(y_true,y_pred,smooth=1e-6):
    print(y_true.get_shape())
    Z, H, W, C = y_true.get_shape().as_list()[1:]
    pred_flat = K.reshape(y_pred, [-1, H * W * Z])
    gt_flat = K.reshape(y_pred, [-1, H * W * Z])
    intersection = K.sum(gt_flat * pred_flat, axis=1)
    union = K.sum(gt_flat, axis=1) + K.sum(gt_flat, axis=1)
    return K.mean( (2. * intersection + smooth) / (union + smooth))
'''

def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)


def iou_loss(in_gt, in_pred):
    return - IoU(in_gt, in_pred)

