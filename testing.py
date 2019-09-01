import tensorflow as tf
from model import Model
from loader import DataLoader
from flags import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FLAGS = tf.app.flags.FLAGS

print('<Creating flags>')
create_flags()

print(f'<Loading data - {FLAGS.dataset}>')
dl = DataLoader(FLAGS.dataset)
dl.load()

print('<Defining model>')
tf.reset_default_graph()
model = Model()

print('<Testing model>')
print(f'Using IBP: {FLAGS.use_ibp}')
print(f'Test epsilon: {FLAGS.test_eps}')
model.test(dl, -1, use_ibp=FLAGS.use_ibp, test_eps=FLAGS.test_eps)
