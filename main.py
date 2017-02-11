import tensorflow as tf
import numpy as np

import ops
from ops import *
from utils import *

class Color():
    def __init__(self):
        self.batch_size = 1
        self.image_size = 256
        self.output_size = 256

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        self.generated_images = self.generator(self.line_images)

        self.real_AB = tf.concat(3, [self.line_images, self.real_images])
        self.fake_AB = tf.concat(3, [self.line_images, self.generated_images])

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.disc_true_logits, tf.ones_like(self.disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.disc_fake_logits, tf.zeros_like(self.disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.disc_fake_logits, tf.ones_like(self.disc_fake_logits))) \
                        + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]


    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)
        e6 = bn(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv')) # e6 is (4 x 4 x self.gf_dim*8)
        e7 = bn(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv')) # e7 is (2 x 2 x self.gf_dim*8)
        e8 = bn(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv')) # e8 is (1 x 1 x self.gf_dim*8)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(3, [d1, e7])
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1), [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(3, [d2, e6])
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2), [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(3, [d3, e5])
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)


c = Color()
