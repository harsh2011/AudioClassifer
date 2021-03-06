# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.keras.utils import CustomObjectScope
import tensorflow as tf
from keras import backend as K

def relu6(x):
  return K.relu(x, max_value=6)

keras_model = "ep-001-vl-1.1738.hdf5"
input_arrays = ["the_input"]
output_arrays = ["the_output"]

# converter = tf.compat.v1.lite.TFLiteConverter
# converter = converter.from_keras_model_file(keras_model, input_arrays,
#                                             output_arrays)
# tflite_model = converter.convert()
# open("converted_speed_keras_model.tflite", "wb").write(tflite_model)
with CustomObjectScope({'relu6': relu6}):
  tflite_model = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_model).convert()
  with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
