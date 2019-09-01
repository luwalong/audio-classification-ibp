import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def create_flags():
    # Data
    tf.app.flags.DEFINE_string('dataset',
            'fsdd',
            'Name of dataset to work on. Now supporting: fsdd')
    tf.app.flags.DEFINE_string('model_name',
            'model',
            'Model name. Saving directory will follow this name')

    # Training parameters
    tf.app.flags.DEFINE_integer('batch_size',
            5,
            'Training batch size')
    tf.app.flags.DEFINE_integer('train_epochs',
            60,
            'Training maximum epochs')
    tf.app.flags.DEFINE_float('learning_rate',
            0.001,
            'Learning rate of Adam optimizer')
    tf.app.flags.DEFINE_boolean('use_ibp',
            False,
            'Train the model with(True)/without(False) IBP')
    
    # Model parameters
    tf.app.flags.DEFINE_float('preemph_coeff',
            0.97,
            'Preemphasizing coefficient (usually: 0.97)')
    tf.app.flags.DEFINE_integer('n_cep',
            22,
            'Number of cepstrum filter')
    tf.app.flags.DEFINE_integer('n_filt',
            16,
            'Number of FFT filter (usually 2^K)')
    tf.app.flags.DEFINE_integer('n_hidden_dim',
            40,
            'Number of hidden dimension in LSTM layers')
    tf.app.flags.DEFINE_integer('frame_size',
            256,
            'Size (number of samples) of a single frame from the raw audio input')
    tf.app.flags.DEFINE_integer('frame_step',
            160,
            'Non-overlapping size of the two consecutive frames')

    # Test arguments
    tf.app.flags.DEFINE_float('test_eps',
            -150.,
            'Perturbing epsilon for the test')

def print_flags():
    print('\nCommand-line Arguments:')
    for key in FLAGS.flag_values_dict():
        print(f'{key.upper(): <22}: {FLAGS[key].value}')

if __name__ == '__main__':
    create_flags()
    print_flags()
