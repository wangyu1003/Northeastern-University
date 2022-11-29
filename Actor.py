import os
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Actor(tf.keras.Model):
    def __init__(self, hparams):
        super(Actor, self).__init__()
        self.hparams = hparams

        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            activation=tf.nn.selu, name="FirstLayer"))

        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def build(self, input_shape=None):
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    @tf.function
    def __call__(self, states_action, states_graph_ids, states_first, states_second, sates_num_edges, training=False):
        link_state = states_action

        # Execute T times
        for _ in range(self.hparams['T']):
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)
            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)
            outputs = self.Message(edgesConcat)
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)

            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]

        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        a = self.Readout(edges_combi_outputs, training=training)
        return a
