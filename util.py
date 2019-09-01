import numpy as np
import tensorflow as tf
import math

class ConstantMatrices():
    def __init__(self, N, s, pc, f, c, sr):
        n = N // 2 + 1
        # Preemphasizing & Hamming
        preemph_mtx = np.array((np.eye(N) - np.eye(N, k=1) * pc) *
                np.hamming(N), dtype=np.float32)
        # Discrete FFT
        W = np.array([
                [[math.cos(i * k * -2. * math.pi / N) for k in range(n)]
                        for i in range(N)],
                [[math.sin(i * k * -2. * math.pi / N) for k in range(n)]
                        for i in range(N)]], 
                dtype=np.float32)
        W_comb = np.concatenate([W[0], W[1]], axis=1) / np.sqrt(N)
        self.aux_mtx = np.matmul(preemph_mtx, W_comb)
        self.fft_add_mtx = np.concatenate([np.eye(n), np.eye(n)], 0) \
                .astype('float32')
        # Mel-frequency Filterbank
        self.fb = self.get_filterbanks(nfilt=f, nfft=N, sr=sr, lf=0, hf=sr/2)
        self.fb = self.fb.T.astype(np.float32)
        # DCT
        W_cos = np.array([[2. / math.sqrt(2. * ((j is 0) + 1) * f) *
                math.cos(math.pi * j * (2. * k + 1.) / (2. * f))
                for j in range(f)] for k in range(f)],
                dtype=np.float32)
        # Lifter
        lifter = np.array(1 + (c/2.) * np.sin(math.pi * np.arange(f) / c),
                dtype=np.float32)
        self.dct_lift = np.matmul(W_cos, np.diag(lifter))

    def mel2hz(self, mel):
        return 700 * (10 ** (mel / 2595.) - 1)

    def hz2mel(self, hz):
        return 2595 * np.log10(1 + hz / 700.)

    def get_filterbanks(self, nfilt, nfft, sr, lf, hf):
        lowmel = self.hz2mel(lf)
        highmel = self.hz2mel(hf)
        melpoints = np.linspace(lowmel, highmel, nfilt+2)
        bin = np.floor((nfft+1) * self.mel2hz(melpoints) / sr)
        fbank = np.zeros([nfilt, nfft//2+1])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return fbank


def overlapping_windows(tensor, size, step, batch):
    '''Split the signal into frames of overlapping windows with given size
    and step. The method is both applicable to original signal and the
    interval bounded signal.

    Args:
        tensor: `Tensor`, with either [?] or [2, ?] shape.
        size: `int`, the length of a single frame.
        step: `int`, stride; the non-overlapping part of two consecutive frames.
        batch: `int`, batch size

    Returns:
        `Tensor` of stacked frames. Note that for 'ibp' mode, return shape
            would be [num_frames, 2, size], otherwise, [num_frames, size]
    '''

    t = tf.reshape(tensor, [batch, -1, 1, 1])
    return tf.squeeze(tf.extract_image_patches(t,
            ksizes=[1, size, 1, 1],
            strides=[1, step, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), axis=2)


def make_interval_bounds(signal, eps):
    '''Make the signal to interval bounds with given perturvation, epsilon.
    Regarding the maximum element of the signal, epsilon is considered as the
    maximum noise difference from the point in decibel(dB) metric. Every
    element in the signal has the bound with the original plus-minus epsilon.
    Note that epsilon is usually negtaive, following the convention written
    on Carlini's paper: https://arxiv.org/pdf/1801.01944.pdf

    Args:
        signal: `Tensor`, windowed input signal.
        eps: `Tensor`, usually negative, coefficient for decibel perturbation.

    Returns:
        Dimension added `Tensor` of perturbation applied signal. [:, 0, ] is
            the lower bound and [:, 1, ] is the upper bound array.
    '''

    dBx = tf.scalar_mul(20. / math.log(10.),
            tf.log(tf.reduce_max(tf.abs(signal), axis=[1, 2])))
    dBd = tf.add(dBx, eps)
    eps_array = tf.pow(10., tf.scalar_mul(0.05, dBd))
    eps_array = tf.expand_dims(tf.expand_dims(eps_array, 1), 2)
    eps_array = tf.tile(eps_array, tf.concat([[1], tf.shape(signal)[1:]], 0))
    return tf.stack([tf.subtract(signal, eps_array), tf.add(signal, eps_array)])


if __name__ == '__main__':
    with tf.Session() as sess:
        demo = np.array([[1, 2, 3, 4]], dtype=np.float32)
        #a = overlapping_windows(demo, 3, 1, 'con')
        #demo = tf.constant([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
        a = overlapping_windows(demo, 3, 1, 1)
        t = sess.run(a)
        print(t.shape)
