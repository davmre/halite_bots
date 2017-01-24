
import datetime
import os
import sys
import socket
import subprocess

import numpy as np
import json
import array

import tensorflow as tf
import tensorflow.contrib.slim as slim


from halite_model import HaliteModel, ModelConfig
import socket_hlt as hlt


HALITE_EXE = "/home/dmoore/python/halite/halite"

def load_models(model_dir, config):
    # given a directory of files containing saved model weights
    # load a list of HaliteModels with those weights? or at least somehow get all the weights into memory
    pass

class DummyModel(object):

    def __init__(self, move_set):
        self.move_set = move_set

    def init_for_map(self, game_map, myID):
        self.myID = myID

    def assign_move(self, square):
        import random
        if square.strength < 5 * square.production:
            return hlt.Move(square, hlt.STILL)
        else:
            return hlt.Move(square, random.choice(self.move_set))

    def preprocess(self, game_map):
        moves = [self.assign_move(square) for square in game_map if square.owner == self.myID]
        return game_map, moves

    def output_moves(self, game_map, moves):
        return moves



def play_game(model1, 
              model2, 
              cmd1, 
              cmd2,
              sock1, 
              sock2):              

    # given two HaliteModels, play a game between them and return the trajectory

    cmd_list = [HALITE_EXE, "-d", "30 30", cmd1, cmd2]
    print(cmd_list)
    p = subprocess.Popen(cmd_list)

    s1, _ = sock1.accept()
    s2, _ = sock2.accept()
    f1 = s1.makefile(mode='rw')
    f2 = s2.makefile(mode='rw')

    hlt.send_init("model1", f1)
    hlt.send_init("model2", f2)

    player1, game_map1 = hlt.get_init(f1)
    player2, game_map2 = hlt.get_init(f2)

    model1.init_for_map(game_map1, player1)
    model2.init_for_map(game_map2, player2)

    trajectory = []

    while True:
        #for game_map, socket, model, player1 in ((game_map1, f1, model1, True), (game_map2, f2, model2, False)):

        try:
            game_map1.get_frame()
        except Exception as e:
            break
        

        state1, action1 = model1.preprocess(game_map1)
        moves1 = model1.output_moves(game_map1, action1)
        hlt.send_frame(moves1, f1)
        trajectory.append((state1, action1))

        game_map2.get_frame()
        state2, action2 = model2.preprocess(game_map2)
        moves2 = model2.output_moves(game_map2, action2)
        hlt.send_frame(moves2, f2)


    f1.close()
    f2.close()
    s1.close()
    s2.close()

    p.wait()

    return trajectory


def update_policy(model, trajectories):
    # given a model and a set of trajectories, train a new model.
    # does not modify the current model in place. 
    raise NotImplementedError()

def save_checkpoint(model, model_dir):
    raise NotImplementedError()

    
def open_sockets(path1 = "./player1", path2 = "./player2"):

    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)

    sock1 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock1.bind(path1)
    sock1.listen(1)

    sock2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock2.bind(path2)
    sock2.listen(1)
    #sock2 = None
    
    cmd1 = 'nc -U %s' % path1
    cmd2 = 'nc -U %s' % path2

    #cmd2 = "python3 MyBotModel.py"

    return sock1, sock2, cmd1, cmd2

"""
def main():
    config = ModelConfig(max_height=48,
                         max_width=48,
                         nwrap=2)

    model_dir = sys.argv[1]
    frozen_models = load_models(model_dir, config)
    tf_session = build_tf_context(config)

    batch_size = 16
    checkpoint_interval = 10

    iters = 0
    while True:
        old_model = sample_model(frozen_models)
        trajectories = play_games(current_model, 
                                  old_model, 
                                  tf_session=tf_session,
                                  batch_size=batch_size)
        current_model = update_policy(current_model, 
                                      trajectories,
                                      tf_session=tf_session)
        frozen_models.append(current_model)
        
        iters += 1
        if (iters % checkpoint_interval) == 0:
            save_checkpoint(current_model, model_dir)
"""

def test_selfplay():

    path1 = "./player1"
    path2 = "./player2"
    sock1, sock2, cmd1, cmd2 = open_sockets(path1=path1, 
                                            path2=path2)

    config = ModelConfig(nwrap=2, max_width=48, max_height=48)

    #model_fname = "train_log.old/model.ckpt-5001"
    model_fname = "expert_games_full.rec_train/model.ckpt-54336" 


    #model1 = DummyModel(move_set=(hlt.NORTH, hlt.EAST))
    model1 = HaliteModel(config)
    model1.register_session(tf.Session(graph=model1.g))
    model1.load_weights_from_file(model_fname)

    model2 = DummyModel(move_set=(hlt.SOUTH, hlt.WEST))
    #model2 = HaliteModel(config)
    #model2.register_session(tf.Session(graph=model2.g))
    #model2.load_weights_from_file(model_fname)

    trajectory = play_game(model1, model2, cmd1, cmd2, sock1, sock2)

    model1.session.close()

    os.remove(path1)
    os.remove(path2)

if __name__ == "__main__":
    test_selfplay()
    
