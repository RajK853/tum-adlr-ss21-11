import tensorflow.compat.v1 as tf


def soft_dice(epsilon=1e-6):
    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        nominator = 2*tf.reduce_sum(y_true*y_pred, axis=axes)
        denominator = tf.math.square(y_true) + tf.math.square(y_pred)
        denominator = tf.reduce_sum(denominator, axis=axes)
        return 1 - tf.reduce_mean((nominator + epsilon)/(denominator + epsilon))
    return loss_func

def tversky(beta=0.25, epsilon=1e-6):
    def loss_func(y_true, y_pred):        
        axes = tf.range(1, tf.rank(y_true))
        nominator = tf.reduce_sum(y_true*y_pred, axis=axes)
        denominator = y_true*y_pred + beta*(1-y_true)*y_pred + (1-beta)*y_true*(1-y_pred)
        denominator = tf.reduce_sum(denominator, axis=axes)
        return 1 - tf.reduce_mean((nominator + epsilon)/(denominator + epsilon))
    return loss_func


def _cross_entropy(w1, w2, y_true, y_pred, epsilon=1e-6):
    return w1*y_true*tf.log(y_pred) + w2*(1-y_true)*tf.log(1-y_pred)


def cross_entropy(beta=1, balanced=False, epsilon=1e-6):
    w1 = beta
    w2 = 1-beta if balanced else 1
    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = _cross_entropy(w1, w2, y_true, y_pred, epsilon=epsilon)
        loss = -tf.reduce_sum(loss, axis=axes)
        return tf.reduce_mean(loss)
    return loss_func


def focal(gamma=2, beta=1.25, balanced=False, epsilon=1e-6):
    beta1 = beta
    beta2 = 1-beta if balanced else 1
    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        w1 = beta1*tf.math.pow(1-y_pred, gamma)
        w2 = beta2*tf.math.pow(y_pred, gamma)
        loss = _cross_entropy(w1, w2, y_true, y_pred, epsilon=epsilon)
        loss = -tf.reduce_sum(loss, axis=axes)
        return tf.reduce_mean(loss)
    return loss_func