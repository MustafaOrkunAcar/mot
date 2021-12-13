import time

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Multiply, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.python import ipu

from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from spektral.layers import ECCConv, AGNNConv, InnerProduct
from spektral.utils import label_to_one_hot
from spektral.data import DisjointLoader

from argparser import get_argparser
from loader import MOTDataset


################################################################################
# PARAMETERS (defaults set in get_argparser())
################################################################################
parser = get_argparser()
args = parser.parse_args()

################################################################################
# Config
################################################################################
learning_rate = args.learning_rate  # Learning rate
epochs = args.epochs  # Number of training epochs
batch_size = args.num_ipus  # Batch size

################################################################################
# CONFIGURE THE DEVICE
################################################################################
cfg = ipu.utils.create_ipu_config(profiling=args.profile,
                                  use_poplar_cbor_report=args.profile,
                                  profile_execution=ipu.utils.ExecutionProfileType.IPU_PROFILE if args.profile else False,
                                  report_directory=args.profile_dir if args.profile else '',
                                  max_report_size=int(5e9))

# auto_select_ipus: specify a quantity of IPUs to use. The virtual device will be given that many IPUs.
#   this causes -> Automatically replicating the TensorFlow model by a factor of $num_ipus.
# select_ipus: allows you to choose a specific IPU hardware device using its ID. 
cfg = ipu.utils.auto_select_ipus(cfg, args.num_ipus)
ipu.utils.configure_ipu_system(cfg)


# tf.keras.backend.set_floatx('float16')->  Mixed precision support ??
# It is not recommended to set this to float16 for training, as this will likely cause numeric stability issues. 
# Instead, mixed precision, which is using a mix of float16 and float32, can be used by calling:
##tf.keras.mixed_precision.set_global_policy('mixed_float16')
# NOTE: this results in compilation errors with pipelined Keras models

tf.keras.mixed_precision.set_global_policy('float32')

################################################################################
# Load data
################################################################################
data_path = "data"    
mot_dataset = MOTDataset(data_path)

for graph in mot_dataset:
    print(graph.a.shape, graph.x.shape, graph.e.shape)


np.random.shuffle(mot_dataset)    
#split = int(0.8 * len(mot_dataset))
#train_data, test_data = mot_dataset[:split], mot_dataset[split:] 
train_data = mot_dataset


# If `node_level=False`, the labels are interpreted as graph-level labels and are stacked along an additional dimension.
# If `node_level=True`, then the labels are stacked vertically.
# note: in fact, we have edge_level labels.
train_loader = DisjointLoader(train_data, node_level=True, batch_size=batch_size, epochs=epochs)
#test_loader = DisjointLoader(test_data, node_level=True, batch_size=batch_size, epochs=1)

################################################################################
# RUN INSIDE OF A STRATEGY
################################################################################
# tf.distribute.Strategy is an API to distribute training across multiple devices. 
# IPUStrategy is a subclass which targets a single system with one or more IPUs attached. 
# Another subclass, IPUMultiWorkerStrategyV1, targets a distributed system with multiple machines (workers). 
# IPU-specific TensorFlow distribution strategy
# tensorflow.python.ipu.ipu_strategy.IPUStrategy is an alias of tensorflow.python.ipu.ipu_strategy.IPUStrategyV1.

##strategy = ipu.ipu_strategy.IPUStrategy()
#cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
#strategy = ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy(cluster_resolver=cluster_resolver)

#def my_net(a):
#  with ipu_scope("/device:IPU:0"):
#    b = a * a
#    with outside_compilation_scope():
#      c = b + 2  # Placed on the host.
#    d = b + c
#    return d




# custom link embedding layer
class Link_Embedding(Layer):

    def __init__(self):
        super(Link_Embedding, self).__init__()

    def call(self, inputs):

        X_2, indices = inputs
        src_nodes, dst_nodes = tf.split(indices, num_or_size_splits=2, axis=1)  # tensors of shape (edge_count, 1)

        src_features = tf.gather_nd(X_2, src_nodes)
        dst_features = tf.gather_nd(X_2, dst_nodes)

        return Concatenate(axis=-1)([src_features, dst_features])


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(24, activation="relu")
        self.conv3 = ECCConv(16, activation="relu")
        self.norm = BatchNormalization()
        self.link_emb = Link_Embedding()
        self.dense = Dense(1, activation="softmax")  # out size 1    # flatten??

    def call(self, inputs):
        x, a, e, i = inputs

        X_1 = self.conv1([x, a, e])
        X_1 = self.norm(X_1)
        X_1 = self.conv2([X_1, a, e]) 
        X_1 = self.conv3([X_1, a, e])
        X_1 = self.link_emb([X_1, a.indices])
        output = self.dense(X_1)
        
        return output  

# best practice: anything intended to be executed on the IPU is placed into a Python function annotated with @tf.function(experimental_compile=True).
# NOTE that this does not apply to constructing a Keras model or using the Keras Model.fit() API. 
# everything within this context will be compiled for the IPU device

#with strategy.scope():
# By default, TensorFlow will create one virtual device (/device:IPU:0) with a single IPU (The first available single IPU).
with ipu.scopes.ipu_scope('/device:IPU:126'):   # using device:IPU:0 causes out of memory error
    ############################################################################
    # BUILD MODEL
    ############################################################################

    # without this I get dim. error. Sizes change.
    tf.config.run_functions_eagerly(True) # tf.executing_eagerly() -> True

    # Create Keras model inside the strategy.
    model = Net()

    optimizer = Adam(lr=learning_rate)

    # gradientDescent is computed once every "count" mini-batches of data.
    ##gradient_accumulation_count = 1
    ##steps_per_execution = args.num_ipus * gradient_accumulation_count

    
    # Compile the model for training.
    # replace loss with tf.nn.weighted_cross_entropy_with_logits
    model.compile(optimizer=optimizer, loss='binary_crossentropy', steps_per_execution=train_loader.steps_per_epoch, run_eagerly=True) #



    ############################################################################
    # FIT MODEL
    ############################################################################

    #train_steps_per_epoch = steps_per_execution if args.profile else train_data_len - train_data_len % steps_per_execution

    tic = time.perf_counter()
    
    #
    loss_fn = BinaryCrossentropy(from_logits=True)

    # experimental_compile=True => Not creating XLA devices, tf_xla_enable_xla_devices not set
    @tf.function(input_signature=train_loader.tf_signature(), experimental_relax_shapes=False, experimental_compile=True)
    def train_step(inputs, target):
        x, a, e, i = inputs
        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    step = 0
    loss = 0
    overall = 0
    losses = []
    for batch in train_loader:
        print("STEP: ", step)
        step += 1
        loss += train_step(*batch)
        if step == train_loader.steps_per_epoch:
            step = 0
            avg_loss = loss / train_loader.steps_per_epoch
            print("Loss: {}".format(avg_loss))
            losses.append(avg_loss)
            loss = 0
    #
    
    toc = time.perf_counter()
    duration = toc - tic
    print(f"Training time duration {duration}")

    print(losses)


    model.summary()

    
    """
    if not args.profile:
        ############################################################################
        # EVALUATE MODEL
        ############################################################################

        print('Testing model')
        model.compile(steps_per_execution=args.num_ipus)
        #test_steps = test_data_len - test_data_len % args.num_ipus

        tic = time.perf_counter()

        model_loss = model.evaluate(test_loader.load(), batch_size=1, steps=1)
        print(f"Done. Test loss {model_loss}")

        toc = time.perf_counter()
        duration = toc - tic
        print(f"Testing time duration {duration}")
    """
    print('Completed')

