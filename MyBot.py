import hlt

import os
import sys
import numpy as np
import tensorflow as tf

from halite_model import HaliteModel, ModelConfig



def main(model_fname):
    hlt.send_init("davmre")

    config = ModelConfig(nwrap=2, max_width=48, max_height=48)
    model = HaliteModel(config)
    
    myID, game_map = hlt.get_init()

    model.init_for_map(game_map, myID)

    with tf.Session(graph=model.g) as sess:
        model.register_session(sess)
        model.load_weights_from_file(model_fname)

        while True:
            game_map.get_frame()
            _, preds = model.preprocess(game_map)
            moves = model.output_moves(game_map, preds)
            hlt.send_frame(moves)

if __name__ == "__main__":
    #model_fname = "train_log.old/model.ckpt-5001"
    model_fname = "expert_games_full.rec_train/model.ckpt-54336" 

    main(model_fname)
