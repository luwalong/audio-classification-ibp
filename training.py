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

print('<Training model>')
print(f'Max Epoch: {FLAGS.train_epochs}')
print(f'Batch Size: {FLAGS.batch_size}')
print(f'Learning Rate: {FLAGS.learning_rate}')
print(f'Using IBP: {FLAGS.use_ibp}')
model.fit(dl, verbose=True, resume=False, use_ibp=FLAGS.use_ibp)
