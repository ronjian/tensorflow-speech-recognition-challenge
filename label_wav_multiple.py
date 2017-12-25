# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python label_wav_multiple.py \
--graph=/home/maikfangogoair/tmp/save/tutorial/conv_graph.pb \
--labels=/home/maikfangogoair/tmp/save/tutorial/conv_labels.txt \
--wave_dir=/home/maikfangogoair/test/audio/ \
--submission_file=/home/maikfangogoair/tmp/save/tutorial/submission.csv 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import io
import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    top_1 = predictions.argsort()[-1]
    human_string = labels[top_1]

    return human_string


def label_wav(wave_dir, labels, graph, input_name, output_name, submission_file):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wave_dir or not tf.gfile.Exists(wave_dir):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)


  wav_list = os.listdir(wave_dir)
  with io.open(file=submission_file, mode='w', encoding='utf-8') as outf:
      for i in range(len(wav_list)):
          if i == 10000 or i==0:
              load_graph(graph)
          with open(wave_dir + wav_list[i], 'rb') as wav_file:
            wav_data = wav_file.read()
            result=run_graph(wav_data, labels_list, input_name, output_name)
            if i == 0:
                outf.write('fname,label\n')
            outf.write(wav_list[i]+','+result+'\n')


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wave_dir, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.submission_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wave_dir', type=str, default='', help='Audio file directory to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--submission_file',
      type=str,
      default='',
      help='Name of submission file name')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
