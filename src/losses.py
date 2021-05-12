import tensorflow.compat.v1 as tf

###################### Cross Entropy Losses ########################

def compute_cross_entropy(w1, w2, y_true, y_pred):
    return w1*y_true*tf.log(y_pred) + w2*(1-y_true)*tf.log(1-y_pred)


def cross_entropy(beta=1, balanced=False, weight=1.0, epsilon=1e-6):
    w1 = beta
    w2 = 1-beta if balanced else 1

    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = compute_cross_entropy(w1, w2, y_true, y_pred)
        loss = -tf.reduce_sum(loss, axis=axes)
        return weight*tf.reduce_mean(loss)
    return loss_func


def focal(gamma=2, beta=0.25, balanced=False, weight=1.0, epsilon=1e-6):
    beta1 = beta
    beta2 = 1-beta if balanced else 1

    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        w1 = beta1*(1-y_pred)**gamma
        w2 = beta2*(y_pred**gamma)
        loss = compute_cross_entropy(w1, w2, y_true, y_pred)
        loss = -tf.reduce_sum(loss, axis=axes)
        return weight*tf.reduce_mean(loss)
    return loss_func


####################### Overlap losses #############################

def soft_dice(weight=1.0, epsilon=1e-6):
    def loss_func(y_true, y_pred):
        axes = tf.range(1, tf.rank(y_true))
        y_true = tf.cast(y_true, dtype=tf.float32)
        nominator = 2*tf.reduce_sum(y_true*y_pred, axis=axes)
        denominator = tf.math.square(y_true) + tf.math.square(y_pred)
        denominator = tf.reduce_sum(denominator, axis=axes)
        loss = 1 - tf.reduce_mean((nominator + epsilon)/(denominator + epsilon))
        return weight*loss
    return loss_func

def tversky(beta=0.25, weight=1.0, epsilon=1e-6):
    def loss_func(y_true, y_pred):        
        axes = tf.range(1, tf.rank(y_true))
        y_true = tf.cast(y_true, dtype=tf.float32)
        nominator = tf.reduce_sum(y_true*y_pred, axis=axes)
        denominator = y_true*y_pred + beta*(1-y_true)*y_pred + (1-beta)*y_true*(1-y_pred)
        denominator = tf.reduce_sum(denominator, axis=axes)
        loss = 1 - tf.reduce_mean((nominator + epsilon)/(denominator + epsilon))
        return weight*loss
    return loss_func

########################## Combine losses ##############################


def combined_loss(loss_configs, weights=None):
    ALL_LOSS_FUNCS = {
        "cross_entropy": cross_entropy,
        "focal": focal,
        "soft_dice": soft_dice,
        "tversky": tversky    
    }

    def init_func(config):
        func_name = config.pop("name")
        func = ALL_LOSS_FUNCS[func_name]
        return func(**config)

    assert len(loss_configs) > 1, f"Expects more than 1 loss function. But received '{loss_funcs}'"
    loss_funcs = [init_func(config) for config in loss_configs]
    weights = [1.0]*len(loss_funcs) if weights is None else weights

    def loss_func(y_true, y_pred):
        losses = [w*func(y_true, y_pred) for w, func in zip(weights, loss_funcs)]
        return tf.reduce_mean(losses)
    return loss_func
