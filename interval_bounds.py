import tensorflow as tf
import math

class IntervalOps():
    def __init__(self):
        pass

    def io_shape(self, shape):
        return shape[:1] + [2] + shape[1:]

    # Generation
    def zeros(self, shape):
        return tf.zeros(self.io_shape(shape))
    def build_itv(self, lb, ub):
        return tf.stack([lb, ub], axis=1)

    # Extraction
    def lb(self, itv):
        return itv[:, 0]
    def ub(self, itv):
        return itv[:, 1]

    # Rearranging modules
    def transpose(self, itv, shape):
        new_shape = [(x + 1 if x > 0 else x) for x in shape]
        return tf.transpose(itv, new_shape[:1]+[1]+new_shape[1:])
    def reshape(self, itv, shape):
        return tf.reshape(itv, self.io_shape(shape))
    def concat(self, itvs, axis):
        return tf.concat(itvs, (axis+1 if axis>0 else axis))
    def split(self, itv, size, axis):
        return tf.split(itv, size, (axis+1 if axis>0 else axis))

    # Monotone functions
    def relu(self, itv):
        return tf.nn.relu(itv)
    def sigmoid(self, itv):
        return tf.sigmoid(itv)
    def tanh(self, itv):
        return tf.tanh(itv)

    # Tricky unary functions
    def log(self, itv):
        return tf.log(itv)
    def square(self, itv):
        zero_not_included = tf.cast(tf.multiply(self.lb(itv), self.ub(itv)) > 0,
                tf.float32)
        lower = tf.minimum(
                tf.minimum(tf.square(self.lb(itv)), tf.square(self.ub(itv))),
                tf.multiply(zero_not_included, tf.float32.max))
        upper = tf.maximum(tf.square(self.lb(itv)), tf.square(self.ub(itv)))
        return self.build_itv(lower, upper)

    # Binary operations
    def affine(self, itv, W, b=None):
        # perform itv * W + b. W and b are constant
        c = tf.add(tf.scalar_mul(0.5, self.lb(itv)),
                tf.scalar_mul(0.5, self.ub(itv)))
        r = tf.add(tf.scalar_mul(-0.5, self.lb(itv)),
                tf.scalar_mul(0.5, self.ub(itv)))

        cW = tf.matmul(c, W)
        rW = tf.matmul(r, tf.abs(W))

        if b == None:
            b = tf.zeros([cW.shape[1]])

        return self.build_itv(
                tf.add(tf.subtract(cW, rW), b),
                tf.add(tf.add(cW, rW), b))

    def add(self, itv1, itv2):
        return tf.add(itv1, itv2)

    def multiply(self, itv1, itv2):
        l1 = self.lb(itv1)
        u1 = self.ub(itv1)
        l2 = self.lb(itv2)
        u2 = self.ub(itv2)
        
        M_ll = tf.multiply(l1, l2)
        M_lu = tf.multiply(l1, u2)
        M_ul = tf.multiply(u1, l2)
        M_uu = tf.multiply(u1, u2)
        
        new_l = tf.minimum(M_ll, tf.minimum(M_lu, tf.minimum(M_ul, M_uu)))
        new_u = tf.maximum(M_ll, tf.maximum(M_lu, tf.maximum(M_ul, M_uu)))
        
        return self.build_itv(new_l, new_u)
