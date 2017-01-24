import tensorflow as tf
import numpy as np
import os

import tensorflow.contrib.slim as slim


def build_model(x):
    print x
    h1 = slim.fully_connected(x, 64, scope='fc/fc_1')
    h2 = slim.fully_connected(h1, 32, scope='fc/fc_2')
    out = slim.fully_connected(h2, 5, activation_fn=None, scope='fc/out')
    w_h = tf.histogram_summary("weights1", h1)
    pred = tf.histogram_summary("pred", out)
    return out


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'board_features': tf.FixedLenFeature((225,), tf.float32),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  feature = features["board_features"]
  label = tf.cast(features['label'], tf.int32)

  return feature, label

def get_filenames():
    basedir =  "/home/dmoore/python/halite/expert_games/hlt_files"
    all_fnames = []
    for player in os.listdir(basedir):
        player_dir = os.path.join(basedir, player)
        fnames = [os.path.join(player_dir, fname) for fname in os.listdir(player_dir) if fname.endswith(".rec")]
        all_fnames += fnames
    return all_fnames

def inputs(batch_size=128, num_epochs=None):

    fnames = get_filenames()

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            fnames, num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, labels


def do_train():

    features, labels = inputs()
    predictions = build_model(features)

    with tf.name_scope("losses_scope"):
        loss = slim.losses.sparse_softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        s1 = tf.summary.scalar('losses/total loss', total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        pred_ints = tf.to_int32(tf.argmax(predictions, dimension=1))
        acc = tf.reduce_mean(tf.to_float(tf.equal(pred_ints, labels)))
        s2 = tf.summary.scalar('accuracy', acc)


    logdir = "/home/dmoore/python/halite/expert_games/log/"
    sess = tf.Session()
    init = tf.initialize_all_variables()
    summary_op = tf.merge_all_summaries()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    writer = tf.train.SummaryWriter(logdir, graph=sess.graph_def)
    
    for i in range(1000):
        _, summary, loss, accv = sess.run((train_op, summary_op, total_loss, acc))
        print "step", i, "loss", loss, "accuracy", accv

        writer.add_summary(summary, i)
        writer.flush()

    #slim.learning.train(
    #    train_op,
    #    logdir,
    #    number_of_steps=1000000,
    #    save_summaries_secs=1,
    #    save_interval_secs=200   )

if __name__=="__main__":
    do_train()
