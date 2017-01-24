import datetime
import os
import sys

import numpy as np
import json
import array

import tensorflow as tf
import tensorflow.contrib.slim as slim


from halite_model import featurize_frame_tf, build_model, wrap_and_pad, ModelConfig


class BoardTooLargeException(Exception):
    pass

def serialize_replay(replay, writer, config, force_target_player=None):

    def _float_feature(arr):
        return tf.train.Feature(float_list=
                                tf.train.FloatList(value=np.float64(arr)))
    def _int_feature(arr):
        return tf.train.Feature(int64_list=
                                tf.train.Int64List(value=[int(x) for x in arr]))
    def _byte_feature(arr):
        barr = array.array('B', arr).tostring()
        return tf.train.Feature(bytes_list=
                                tf.train.BytesList(value=barr))

    width, height = replay["width"], replay["height"]

    if width > config.max_width-2*config.nwrap or \
       height > config.max_height-2*config.nwrap:
        raise BoardTooLargeException("board size %d x %d does not match target %d x %d, skipping..." % (width, height, config.max_width, config.max_height))

    # get the id of the winning player (most territory in final frame)
    frames = np.asarray(replay["frames"])
    moves = np.asarray(replay["moves"])


    if force_target_player is not None:
        target_id = None
        for i, pname in enumerate(replay["player_names"]):
            if force_target_player in pname:
                target_id = i + 1
                break
        if target_id is None:
            raise Exception("could not find forced player %s in list %s" % (force_target_player, replay["player_names"]))
    else:
        # target the winning player
        players,counts = np.unique(frames[-1,:, :, 0],return_counts=True)
        # take the argmax only over actual players, not blank spaces
        target_id = players[counts[1:].argmax()] + 1
        
    nframes = len(frames)-1

    players = wrap_and_pad(frames[:, :, :, 0], config)
    strengths = wrap_and_pad(frames[:, :, :, 1], config)
    productions = wrap_and_pad(np.asarray(replay["productions"]), config)
    moves = wrap_and_pad(moves, config, nwrap=0)

    if (moves >= 5).any():
        raise Exception("invalid move %d, skipping" % np.max(moves))
    if (strengths >= 256).any():
        raise Exception("invalid strength %f, skipping" % np.max(strengths))

    prod_feature = _int_feature(productions.flatten())
    for frame in range(nframes):
        player = players[frame, :, :].flatten()
        strength = strengths[frame, :, :].flatten() 
        move = moves[frame, :, :].flatten()
        feature={'player': _int_feature(player),
                 'strength': _float_feature(strength),
                 'production': prod_feature,
                 'moves': _int_feature(move),
                 'height': _int_feature([height,]),
                 'width': _int_feature([width,]),
                 'frame': _int_feature([frame,]),
                 'target_id': _int_feature([target_id])}

        Features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=Features)
        writer.write(example.SerializeToString())


def replays_to_tfrecord(replay_folder, record_fname, 
                        config,
                        force_target_player=None,
                        max_replays=3):

    # save all replays to a tensorflow record file for easy loading

    size = len(os.listdir(replay_folder))    

    writer = tf.python_io.TFRecordWriter(record_fname)
    for index, replay_name in enumerate(os.listdir(replay_folder)[:max_replays]):

        if replay_name[-4:]!='.hlt':continue
        print('Loading {} ({}/{})'.format(replay_name, index, size))
        replay = json.load(open('{}/{}'.format(replay_folder,replay_name)))

        try:
            serialize_replay(replay, writer, 
                             force_target_player=force_target_player,
                             config=config)
        except Exception as e:
            print(e)
            continue

    writer.close()

def deserialize_record(serialized, config):

  record_size = np.prod(config.shape())

  features = tf.parse_single_example(
      serialized,
      # Defaults are not specified since both keys are required.
      features={
          'player': tf.FixedLenFeature((record_size,), tf.int64),
          'strength': tf.FixedLenFeature((record_size,), tf.float32),
          'production': tf.FixedLenFeature((record_size,), tf.int64),
          'moves': tf.FixedLenFeature((record_size,), tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'frame': tf.FixedLenFeature([], tf.int64),
          'target_id': tf.FixedLenFeature([], tf.int64),
      })

  frame = features["frame"]
  player = tf.reshape(features["player"], config.shape())
  strength = tf.reshape(features["strength"], config.shape())
  production = tf.reshape(features["production"], config.shape())
  moves = tf.reshape(features["moves"], config.shape())

  board, weights = featurize_frame_tf(player,
                                      strength,
                                      production,
                                      features["target_id"])

  moves = tf.slice(moves, (0, 0), (config.max_height-2*config.nwrap, 
                                   config.max_width-2*config.nwrap))
  weights = tf.slice(weights, (config.nwrap, config.nwrap), 
                     (config.max_height-2*config.nwrap, 
                      config.max_width-2*config.nwrap))

  return board, moves, weights, frame

def inputs_from_records(record_fname, config, batch_size=32):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [record_fname,], capacity=1)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        board, moves, ws, _ = deserialize_record(serialized_example, 
                                                 config=config)

        boards, targets, weights = tf.train.shuffle_batch(
            [board, moves, ws], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return boards, targets, weights


def do_train(record_fname, config, logdir=None):

    inputs, targets, weights = inputs_from_records(record_fname, 
                                                   config=config)

    preds = build_model(inputs)

    with tf.name_scope("losses"):

        mask = tf.squeeze(tf.slice(inputs, (0, config.nwrap, config.nwrap, 1), 
                        (-1, config.max_height-2*config.nwrap, 
                         config.max_width-2*config.nwrap, 1)))


        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, 
                                                            logits=preds)
        #weights2 = 9 * tf.to_float(targets > 0)  + 1
        #weights *= weights2

        ce_loss = tf.reduce_sum(ce * weights)
        slim.losses.add_loss(ce_loss)
        total_loss = slim.losses.get_total_loss()
        
        baseline_preds = tf.zeros_like(preds)
        baseline_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, 
                                                                     logits=baseline_preds)
        baseline_loss = tf.reduce_sum(baseline_ce * weights)

        s1 = tf.summary.scalar('losses/total loss', total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        hard_preds = tf.argmax(preds, dimension=3)
        correct = tf.to_float(tf.equal(targets, hard_preds))

        accuracy = tf.reduce_sum(correct*mask) / tf.reduce_sum(mask)
        s2 = tf.summary.scalar('losses/accuracy', accuracy)

        stills_mask = tf.to_float(tf.equal(targets, 0)) * mask
        moves_mask = tf.to_float(tf.not_equal(targets, 0)) * mask
        stills_accuracy = tf.reduce_sum(correct*stills_mask) / tf.reduce_sum(stills_mask + 1e-8)
        moves_accuracy = tf.reduce_sum(correct*moves_mask) / tf.reduce_sum(moves_mask + 1e-8)
        s3 = tf.summary.scalar('losses/stills_accuracy', stills_accuracy)
        s4 = tf.summary.scalar('losses/moves_accuracy', moves_accuracy)

        baseline_correct = tf.to_float(tf.equal(targets, 0))
        baseline_accuracy = tf.reduce_sum(baseline_correct*mask) / tf.reduce_sum(mask)

        improvement = accuracy - baseline_accuracy
        s3 = tf.summary.scalar('losses/improvement', improvement)


    if logdir is None:
        logdir = record_fname + "_train"


    slim.learning.train(train_op, logdir, 
                        number_of_steps=200000,
                        save_summaries_secs=2,
                        save_interval_secs=30)


def main():
    #verify_serialization("../replays/ar1481914392-351579806.hlt")

    config = ModelConfig(max_height=56,
                         max_width=56,
                         nwrap=3)


    record_fname = sys.argv[1]
    tgt = "erdman"

    if not os.path.exists(record_fname):
        replay_folder = sys.argv[2]
        replays_to_tfrecord(replay_folder, record_fname, 
                            max_replays=5000,
                            config=config,
                            force_target_player=tgt)

    do_train(record_fname, 
             logdir="erdman_bignet",
             config=config)


if __name__ == "__main__":
    main()
