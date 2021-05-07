import tensorflow.compat.v1 as tf


def soft_dice_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    y_true = tf.reshape(y_true, [batch_size, -1])
    y_pred = tf.reshape(y_pred, [batch_size, -1])
    nominator = 2*tf.reduce_sum(y_true*y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true*y_true, axis=-1)+tf.reduce_sum(y_pred*y_pred, axis=-1)
    loss = tf.reduce_mean(1-nominator/denominator)
    return loss