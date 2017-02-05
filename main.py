import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os

class Colorize():
    def __init__(self):

        self.img_size = 512
        self.num_colors = 3

        self.batch_size = 1

        self.images_in = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])
        self.images_out = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        # breaking down the context
        h0 = lrelu(conv2d(self.images_in, self.num_colors, 64, name='d_h0_conv')) #256x256x64
        h1 = lrelu(d_bn1(conv2d(h0, 64, 128, name='d_h1_conv'))) #128x128x128
        h2 = lrelu(d_bn2(conv2d(h1, 128, 256, name='d_h2_conv'))) #64x64x256

        print h2.get_shape()

        # generating the new replacement
        h3 = tf.nn.relu(g_bn1(conv_transpose(h2, [self.batch_size, 128, 128, 128], "g_h3"))) #128x128x128
        h4 = tf.nn.relu(g_bn2(conv_transpose(h3, [self.batch_size, 256, 256, 64], "g_h4"))) #256x256x64
        self.generated_images = tf.nn.tanh(g_bn3(conv_transpose(h4, [self.batch_size, 512, 512, 3], "g_h6"))) #512x512x3

        self.generation_loss = tf.nn.l2_loss(self.images_out - self.generated_images)

        self.cost = self.generation_loss
        optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def train(self):

        data = glob(os.path.join("imgs", "*.jpg"))
        base = np.array([get_image(sample_file, 512, is_crop=False) for sample_file in data[0:64]])
        base += 1
        base /= 2

        print base.shape
        ims("results/base.jpg",merge(base, [8,8]))

        for e in xrange(20000):
            for i in range(len(data) / self.batch_size):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file, 512, is_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2

                fill, gen_loss, _ = self.sess.run([self.generated_images, self.generation_loss, self.train_op], feed_dict={self.images_in: batch_images, self.images_out: batch_images})
                print "iter %d genloss %f" % (e, gen_loss)
                if e % 3 == 0:
                    recreation = fill
                    ims("results/"+str(e)+".jpg",recreation[0])



model = Colorize()
model.train()
