import numpy as np
import tensorflow as tf
import json
import os

from train_supervised import deserialize_record
from halite_model import ModelConfig, wrap_and_pad

def main(record_fname, config):


    filename_queue = tf.train.string_input_producer(
        [record_fname,], capacity=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    board, moves, ws, frame = deserialize_record(serialized_example, 
                                             config=config)


    sess = tf.Session()
    init = tf.initialize_all_variables()    
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    i = 0
    while True:
        board_val, moves_val, ws_val, frame_val = sess.run((board, moves, ws, frame))

        assert((moves_val < 5).all())

        i += 1
        if i % 1000 == 0:
            print("verified %d" % i)

def check_replays(replay_folder):

    config = ModelConfig(max_height=56,
                         max_width=56,
                         nwrap=3)


    size = len(os.listdir(replay_folder))
    for index, replay_name in enumerate(os.listdir(replay_folder)[800:]):

        if replay_name[-4:]!='.hlt':continue
        print('Loading {} ({}/{})'.format(replay_name, index, size))
        replay = json.load(open('{}/{}'.format(replay_folder,replay_name)))
        moves = np.asarray(replay["moves"])

        moves_padded = wrap_and_pad(moves, config, nwrap=0)

        assert((moves < 5).all())
        assert((moves_padded < 5).all())


if __name__ == "__main__":

    check_replays("../expert_games/hlt_files/erdman/")

    """
    config = ModelConfig(max_height=56,
                         max_width=56,
                         nwrap=3)

    record_fname = "erdman.rec"

    main(record_fname, config)


    """
