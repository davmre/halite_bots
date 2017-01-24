import numpy as np
import hlt

import tensorflow as tf
import tensorflow.contrib.slim as slim

class ModelConfig(object):
    
    def __init__(self, max_height=48, max_width=48, nwrap=2):
        self.max_height = max_height
        self.max_width = max_width
        self.nwrap = nwrap
        
    def shape(self):
        return (self.max_height, self.max_width)

class HaliteModel(object):
    
    def __init__(self, config):
        self.config = config

        self.g = tf.Graph()
        i, p, s, o, f, m = self._setup_model(self.g)
        self.tf_id = i
        self.tf_production = p
        self.tf_strengths = s
        self.tf_owners = o
        self.features = f
        self.model = m

        with self.g.as_default():
            self.restorer = tf.train.Saver()
        self.session = None
        
    def load_weights_from_file(self, model_fname):
       self.restorer.restore(self.session, model_fname)

    def serialize_weights_to_file(self, model_file):
        raise NotImplementedError()

    def register_session(self, sess):
        assert(self.session is None)
        self.session = sess

    def init_for_map(self, game_map, myID):
        self.height = game_map.height
        self.width = game_map.width
        self.production = wrap_and_pad(np.asarray(game_map.production), 
                                  config=self.config)
        self.myID = myID

    def _setup_model(self, g):
        with g.as_default():
            tf_id = tf.placeholder(shape=(), dtype=tf.float32)
            tf_production = tf.placeholder(shape=self.config.shape(), dtype=tf.float32)
            tf_strengths = tf.placeholder(shape=self.config.shape(), dtype=tf.float32)
            tf_owners = tf.placeholder(shape=self.config.shape(), dtype=tf.float32)

            features, weights = featurize_frame_tf(tf_owners, tf_strengths, tf_production, tf_id)
            model = tf.squeeze(build_model(tf.expand_dims(features, axis=0)))

        return tf_id, tf_production, tf_strengths, tf_owners, features, model

    def preprocess(self, game_map):
        strengths, owners = unpack_squares(game_map, self.config)
        fd = {self.tf_id: self.myID,
              self.tf_production: self.production,
              self.tf_strengths: strengths, 
              self.tf_owners: owners}
        assert(self.session is not None)

        features, preds = self.session.run((self.features, self.model), feed_dict=fd)
        return features, preds

    def output_moves(self, game_map, preds):
        moves = sample_gumbel_softmax(preds)
        return moves_to_hlt(moves, game_map, self.myID)



def unpack_squares(game_map, config):
    strengths = np.empty((game_map.height, game_map.width))
    owners = np.empty((game_map.height, game_map.width), dtype=np.int64)

    for square in game_map:
        strengths[square.y, square.x] = square.strength
        owners[square.y, square.x] = square.owner

    wstrengths = wrap_and_pad(strengths, config)
    wowners = wrap_and_pad(owners, config)
    return wstrengths, wowners


def sample_gumbel_softmax(preds):
    # given unnormalized log-probabilities for each move,
    # sample concrete moves
    # using the gumbel-max trick
    u = np.random.rand(*preds.shape)
    gumbels = -np.log(-np.log(u))
    moves = np.argmax(preds+gumbels, axis=-1)

    return moves


def moves_to_hlt(moves, game_map, myID):
    # convert a numpy array of shape (height, width) containing integer move ids, 
    # into a list of Move objects for passing back to the game engine.

    # integer mappings in replay files don't match those defined in hlt.py.
    # these are the version from the replay files (thus the model)
    move_consts = {0: hlt.STILL, 
                   1: hlt.NORTH, 
                   2: hlt.EAST, 
                   3: hlt.SOUTH, 
                   4: hlt.WEST}

    hlt_moves = [hlt.Move(square, move_consts[moves[square.y, square.x]]) for square in game_map if square.owner == myID]

    return hlt_moves

def wrap_and_pad(board, config, nwrap=None):

    if nwrap is None:
        nwrap = config.nwrap

    # assume height, width are the final two dimensions,
    # and don't pad any of the others.
    height, width = board.shape[-2:]
    base_padding = [(0, 0) for i in board.shape[:-2]]

    wrapping = base_padding + [(nwrap, nwrap), 
                               (nwrap, nwrap)]
    wrapped_board = np.pad(board, wrapping, 'wrap')

    pad_height = config.max_height - wrapped_board.shape[-2]
    pad_width = config.max_width - wrapped_board.shape[-1]
    padding = base_padding + [(0, pad_height), (0, pad_width)]
    padded_board = np.pad(wrapped_board, padding, 'constant')

    return padded_board


def featurize_frame_tf(player, strength, production, target_player):

    f1 = tf.cast(tf.equal(player, 0), tf.float32)
    f2 = tf.cast(tf.equal(player, target_player), tf.float32)
    f3 = (1-f1) * (1-f2) # not unoccupied and not occupied by us, thus
                         # occupied by an opponent
    
    scaled_strength = tf.cast(strength, tf.float32) / 20.0
    scaled_prod = tf.cast(production, tf.float32) / 255.0

    features = tf.pack((f1, f2, f3, f1*scaled_strength, f2*scaled_strength, f3*scaled_strength, scaled_strength, scaled_prod), axis=-1)

    target_squares = f2
    target_weights = 1.0/(tf.reduce_sum(target_squares) + 1e-6)
    weights = target_squares * target_weights
    
    return features, weights


"""
def build_model(inputs):

    with tf.name_scope("forward_model"):
        net = slim.conv2d(inputs, 64, [5, 5], 
                          activation_fn=tf.nn.relu,
                          scope='conv1', 
                          padding='VALID')


        net = slim.conv2d(net, 16, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv2', 
                          padding='SAME')

        net = slim.conv2d(net, 5, [1, 1],
                          scope='conv3',
                          activation_fn=None,
                          padding='SAME')

    return net
"""

"""
def build_model(inputs):
    
    with tf.name_scope("forward_model"):
        net = slim.conv2d(inputs, 64, [5, 5], 
                          activation_fn=tf.nn.relu,
                          scope='conv1', 
                          padding='VALID')


        net = slim.conv2d(net, 32, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv3', 
                          padding='SAME')

        net = slim.conv2d(net, 32, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv4', 
                          padding='SAME')

        net = slim.conv2d(net, 16, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv5', 
                          padding='SAME')

        net = slim.conv2d(net, 5, [1, 1],
                          scope='conv6',
                          activation_fn=None,
                          padding='SAME')
    return net
"""

def build_model(inputs):
    
    with tf.name_scope("forward_model"):
        net = slim.conv2d(inputs, 64, [7, 7], 
                          activation_fn=tf.nn.relu,
                          scope='conv1', 
                          padding='VALID')


        net = slim.conv2d(net, 64, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv3', 
                          padding='SAME')

        net = slim.conv2d(net, 64, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv4', 
                          padding='SAME')

        net = slim.conv2d(net, 32, [3, 3], 
                          activation_fn=tf.nn.relu,
                          scope='conv5', 
                          padding='SAME')

        net = slim.conv2d(net, 5, [1, 1],
                          scope='conv6',
                          activation_fn=None,
                          padding='SAME')
    return net
