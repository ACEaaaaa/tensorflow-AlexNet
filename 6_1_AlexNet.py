#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
from datetime import datetime
import math
import time
import tensorflow as tf

#测试100个batch的数据，batch  size=32
batch_size=32
num_batches=100

#接受一个tensor作为输入 并且显示名称t.op.name 和 tensor尺寸t.get_shape().as_list()
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        #初始化卷积核参数kernel
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        #print(images)
        #卷积操作
        #image size[224,224,3]
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        #conv1    [32,56,56,64]
        
        #偏置项
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        
        #
        bias = tf.nn.bias_add(conv, biases)
        
        #使用激活函数对结果进行非线性处理
        conv1 = tf.nn.relu(bias, name=scope)
        
        print_activations(conv1)
        #用11*11*3的卷积核卷积224*224*3的图像，卷积后的尺寸是56*56*1。
        #conv1   [32, 56, 56, 64]
        
        parameters += [kernel, biases]


  # pool1
       
  lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')#VALID 取样时不能超过边框 ,即取样时不能超过边框，
    #不像SAME模式那样可以填充边界外的点
    print_activations(pool1)
    #pool1   [32, 27, 27, 64] 

  # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        #pool1   [32, 27, 27, 64] 
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        #SAME模式可以填充边界点
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)
  #  conv2   [32, 27, 27, 192]


  # pool2
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)
     #pool2   [32, 13, 13, 192]
    #pool2   [32, 13, 13, 192]

  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        #pool2   [32, 13, 13, 192]
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        #conv3   [32, 13, 13, 384]
        parameters += [kernel, biases]
        print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        #conv4   [32, 13, 13, 256]
        parameters += [kernel, biases]
        print_activations(conv4)
        #conv4   [32, 13, 13, 256]
  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
         #conv5   [32, 13, 13, 256]

  # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)
    #pool5   [32, 6, 6, 256]
    return pool5, parameters


def time_tensorflow_run(session, target, info_string):
#  """Run the computation to obtain the target tensor and print timing stats.
#
#  Args:
#    session: the TensorFlow session to run the computation under.
#    target: the target Tensor that is passed to the session's run() function.
#    info_string: a string summarizing this run, to be printed with the stats.
#
#  Returns:
#    None
#  """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))



def run_benchmark():
#  """Run the benchmark on AlexNet."""
    with tf.Graph().as_default():
    # Generate some dummy images.
        image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
        pool5, parameters = inference(images)

    # Build an initialization operation.
        init = tf.global_variables_initializer()

    # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        sess.run(init)

    # Run the forward benchmark.
        time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
        objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
        grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()

