#!/usr/bin/env python3
import argparse
import os
import sys
from keras.models import load_model, model_from_json
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import gfile
import tensorflow as tf


def convert_keras_to_pb(model, out_names, out_model_path):
    names = ",".join(x.name[:x.name.rindex(":")] for x in model.outputs)
    if out_names is None:
        out_names = names
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    try:
        os.mkdir("saved_ckpt")
    except OSError:
        pass
    checkpoint_path = saver.save(sess, "./saved_ckpt", global_step=0, latest_filename="checkpoint_state")
    graph_io.write_graph(sess.graph, ".", "tmp.pb")
    freeze_graph.freeze_graph(
        "./tmp.pb", "", False, checkpoint_path, out_names,
        "save/restore_all", "save/Const:0",
        out_model_path, False, "")
    print("Saved output to", out_model_path)
    #load freeze graph
    with gfile.FastGFile('./model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()    
        tf.import_graph_def(graph_def)
    n = tf.get_default_graph().as_graph_def().node
    print('use this as input name: ' + n[0].name)
    print('use this as output name: ' + n[-1].name.replace('import/', ''))


class Main(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", type=str, default="", help="HDF5 file with keras model & weights")
        parser.add_argument("-j", "--json", type=str, default="", help="JSON file with keras model")
        parser.add_argument("-w", "--weights", type=str, default="", help="HDF5 file with keras weights")
        args = parser.parse_args()

        if not len(args.model) and (not len(args.json) or not len(args.weights)):
            print("Either --model is required or (--json and --weights)")
            sys.exit(1)

        K.set_learning_phase(0)
        if args.model:
            model = load_model(args.model)
        else:
            with open(args.json, "r") as fin:
                jdata = fin.read()
                model = model_from_json(jdata)
            model.load_weights(args.weights)

        print("model.input_shape:", model.input_shape)

        convert_keras_to_pb(model, None, "./model.pb")


if __name__ == "__main__":
    Main()
    print("Done")
