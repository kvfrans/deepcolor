import cv2
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import ops

class Colorize():
    def __init__(self):

        self.img_size = 256
        self.num_colors = 3
        self.batch_size = 9
        self.batch_size_sqrt = 3

        # IMPORTANT: comments are based on the context of a 512x512 image. Code *should* work on other powers of 2 as well.

        # in both discriminator and generator, we compress an image into a 64x64x256 matrix of features
        self.bridge_size = (self.img_size/8) * (self.img_size/8) * 128 # 64*64*256

        def generator():
            with tf.variable_scope("generator"):
                h0 = lrelu(conv2d(self.images_in, 1, 64, name='g_h0_conv')) #256x256x64
                h1 = lrelu(bn(conv2d(h0, 64, 128, name='g_h1_conv'))) #128x128x128
                h2 = lrelu(bn(conv2d(h1, 128, 128, name='g_h2_conv'))) #self.bridge_size
                h3 = tf.nn.relu(bn(conv_transpose(h2, [self.batch_size, self.img_size/4, self.img_size/4, 128], "g_h3"))) #128x128x128
                h4 = tf.nn.relu(bn(conv_transpose(h3, [self.batch_size, self.img_size/2, self.img_size/2, 64], "g_h4"))) #256x256x64
                return tf.nn.tanh(bn(conv_transpose(h4, [self.batch_size, self.img_size, self.img_size, 3], "g_h6"))) #512x512x3

        d_bn1 = batch_norm(name="d_bn1")
        d_bn2 = batch_norm(name="d_bn2")
        d_bn3 = batch_norm(name="d_bn3")
        d_bn4 = batch_norm(name="d_bn4")

        def discriminator(image, reuse=False):
            with tf.variable_scope("discriminator"):

                if reuse:
                    tf.get_variable_scope().reuse_variables()

                # process given image
                h0 = lrelu(conv2d(image, 3, 64, name='d_h0_conv')) #256x256x64
                h1 = lrelu(d_bn1(conv2d(h0, 64, 128, name='d_h1_conv'))) #128x128x128
                h2 = lrelu(d_bn2(conv2d(h1, 128, 128, name='d_h2_conv'))) #64x64x256

                # process line image
                m0 = lrelu(conv2d(self.images_in, 1, 64, name='d_m0_conv')) #256x256x64
                m1 = lrelu(d_bn3(conv2d(m0, 64, 128, name='d_m1_conv'))) #128x128x128
                m2 = lrelu(d_bn4(conv2d(m1, 128, 128, name='d_m2_conv'))) #64x64x256

                flattened = tf.concat(1, [tf.reshape(h2, [self.batch_size, self.bridge_size]), tf.reshape(m2, [self.batch_size, self.bridge_size])])
                h3 = lrelu(dense(flattened, self.bridge_size * 2, 64, "d_h3"))
                return tf.nn.sigmoid(dense(h3, 64, 1))

        self.images_in = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.images_out = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.num_colors])

        self.generated_images = generator()
        self.discriminator_generated = discriminator(self.generated_images)
        self.discriminator_true = discriminator(self.images_out, reuse=True)

        # Discriminator wants: correctly identify true and gen images
        d_true_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.discriminator_true, tf.ones_like(self.discriminator_true)))
        d_gen_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.discriminator_generated, tf.zeros_like(self.discriminator_generated)))
        self.d_loss = d_true_loss + d_gen_loss

        # Generator wants: trick discriminator into accepting its images
        self.g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.discriminator_generated, tf.ones_like(self.discriminator_generated)))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        print [var.name for var in d_vars]
        print [var.name for var in g_vars]

        self.d_optim = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list=g_vars)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def train(self):

        data = glob(os.path.join("imgs", "*.jpg"))
        base = np.array([get_image(sample_file) for sample_file in data[0:self.batch_size]])
        base_normalized = base/255.0

        base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in base]) / 255.0
        base_edge = np.expand_dims(base_edge, 3)

        ims("results/base.png",merge_color(base_normalized, [self.batch_size_sqrt,self.batch_size_sqrt]))
        ims("results/base_line.jpg",merge(base_edge, [self.batch_size_sqrt,self.batch_size_sqrt]))

        for e in xrange(20000):
            for i in range(len(data) / self.batch_size):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = np.array([get_image(batch_file) for batch_file in batch_files])
                batch_normalized = batch/255.0

                batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
                batch_edge = np.expand_dims(batch_edge, 3)

                d_loss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.images_in: batch_edge, self.images_out: batch_normalized})
                g_loss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.images_in: batch_edge})
                # g_loss = 0

                print "epoch %d iter %d d_loss %f g_loss %f" % (e, i, d_loss, g_loss)
                if i % 50 == 0:

                    recreation = self.sess.run(self.generated_images, feed_dict={self.images_in: batch_edge, self.images_out: batch_normalized})
                    recreation_base = self.sess.run(self.generated_images, feed_dict={self.images_in: base_edge, self.images_out: base_normalized})

                    ims("results/"+str(e*100000 + i)+"-line.jpg",merge(batch_edge, [self.batch_size_sqrt,self.batch_size_sqrt]))
                    ims("results/"+str(e*100000 + i)+".jpg",merge_color(recreation, [self.batch_size_sqrt,self.batch_size_sqrt]))
                    ims("results/"+str(e*100000 + i)+"-base.jpg",merge_color(recreation_base, [self.batch_size_sqrt,self.batch_size_sqrt]))



model = Colorize()
model.train()
