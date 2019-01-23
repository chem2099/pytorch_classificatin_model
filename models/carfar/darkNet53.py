import tensorflow as tf
from models.config import config  as cfg

__all__ = ['darknet53']


class DarkNet53:
    def __init__(self, inputs, training):
        # net config
        self.num_class = cfg.NUM_CLASS
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.scale = self.image_size / self.cell_size
        self.num_anchors = cfg.NUM_ANCHORS
        self.anchors = cfg.ANCHORS
        self.anchor_mask = cfg.ANCHOR_MASK
        self.x_scale = cfg.X_SCALE
        self.y_scale = cfg.Y_SCALE
        self.drop_rate = cfg.DROP_RATE

        # _Leaky_Relu config
        self.alpha = cfg.ALPHA

        self.inputs = inputs

        self.training = training

        self._feature_extractor_layer(inputs, training)
        self._detection_layer(training)

    def _feature_extractor_layer(self, inputs, training):
        with tf.variable_scope('visualization'):
            tf.summary.image('image', inputs, max_outputs=255)

        with tf.variable_scope('darknet_53'):
            # with tf.device('/gpu:0'):
            num_layer = 0

            # 0
            layer = self._Conv2d(inputs, filters=32, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1

            # 1
            layer = self._Conv2d(layer, filters=64, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            # 2 - 4
            for _ in range(1):
                layer = self._Conv2d(layer, filters=32, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Conv2d(layer, filters=64, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Residual(layer, shortcut)

            # 5
            layer = self._Conv2d(layer, filters=128, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            # 6 - 9
            for _ in range(2):
                layer = self._Conv2d(layer, filters=64, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Conv2d(layer, filters=128, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Residual(layer, shortcut)

            # 10
            layer = self._Conv2d(layer, filters=256, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer10_image = layer[0:1, :, :, 0:256]
                layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer10_image, max_outputs=255)

            # 11 - 26
            for _ in range(8):
                layer = self._Conv2d(layer, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Conv2d(layer, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Residual(layer, shortcut)
            self.scale_2 = layer

            # 27
            layer = self._Conv2d(layer, filters=512, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer27_image = layer[0:1, :, :, 0:512]
                layer27_image = tf.transpose(layer27_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer27_image, max_outputs=255)

            # 28 - 44
            for _ in range(8):
                layer = self._Conv2d(layer, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Conv2d(layer, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Residual(layer, shortcut)
            self.scale_1 = layer

            # 45
            layer = self._Conv2d(layer, filters=1024, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                 training=training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer45_image = layer[0:1, :, :, 0:1024]
                layer45_image = tf.transpose(layer45_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer45_image, max_outputs=255)

            # 46 - 53
            for _ in range(4):
                layer = self._Conv2d(layer, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Conv2d(layer, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                     training=training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self._Residual(layer, shortcut)
            self.scale_0 = layer

    def _detection_layer(self, training=True):
        with tf.name_scope('detection_layer'):
            with tf.variable_scope('scale_0'):
                # with tf.device('/gpu:0'):
                self.scale_0 = self._Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_1')
                self.scale_0 = self._Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_2')
                self.scale_0 = self._Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_0 = self._Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_4')
                self.scale_0 = self._Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_5')
                layer_final = self._UpSampling2d(self.scale_0, 256, shape=[1, 1], strides=(2, 2),
                                                 name='_UpSampling2d')
                self.scale_0 = self._Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_6')
                self.scale_0 = self._Conv2d(self.scale_0,
                                            filters=(self.num_class + 5) * self.num_anchors,
                                            shape=[1, 1],
                                            stride=(1, 1),
                                            alpha=self.alpha,
                                            training=training,
                                            name='_Conv2d_output')

            with tf.variable_scope('scale_1'):
                # with tf.device('/gpu:0'):
                self.scale_1 = self._Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_1')
                self.scale_1 = tf.concat([self.scale_1, layer_final], 3, name='concat_scale_0_to_scale_1')
                self.scale_1 = self._Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_2')
                self.scale_1 = self._Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_1 = self._Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_4')
                self.scale_1 = self._Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_5')
                layer_final = self._UpSampling2d(self.scale_1, 128, shape=[1, 1], strides=(2, 2),
                                                 name='_UpSampling2d')
                self.scale_1 = self._Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_6')
                self.scale_1 = self._Conv2d(self.scale_1,
                                            filters=(self.num_class + 5) * self.num_anchors,
                                            shape=[1, 1],
                                            stride=(1, 1),
                                            alpha=self.alpha,
                                            training=training,
                                            name='_Conv2d_output')

            with tf.variable_scope('scale_2'):
                # with tf.device('/gpu:0'):
                self.scale_2 = self._Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_1')
                self.scale_2 = tf.concat([self.scale_2, layer_final], 3,
                                         name='concat_scale_1_to_scale_2')
                self.scale_2 = self._Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_2')
                self.scale_2 = self._Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_2 = self._Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_4')
                self.scale_2 = self._Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_5')
                self.scale_2 = self._Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                            training=training, name='_Conv2d_6')
                self.scale_2 = self._Conv2d(self.scale_2,
                                            filters=(self.num_class + 5) * self.num_anchors,
                                            shape=[1, 1],
                                            stride=(1, 1),
                                            alpha=self.alpha,
                                            training=training,
                                            name='_Conv2d_output')

    def _Leaky_Relu(self, input, alpha=0.01):
        output = tf.maximum(input, tf.multiply(input, alpha))

        return output

    def _Conv2d(self, inputs, filters, shape, stride=(1, 1),
                alpha=0.01, is_drop_out=False, is_batch_normal=True, is_Leaky_Relu=True,
                training=True, name=None):
        layer = tf.layers.conv2d(inputs,
                                 filters,
                                 shape,
                                 stride,
                                 padding='SAME',
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32),
                                 name=name)
        if is_drop_out:
            layer = tf.layers.dropout(layer, self.drop_rate, training=training)

        if is_batch_normal:
            layer = tf.layers.batch_normalization(layer, training=training)

        if is_Leaky_Relu:
            layer = self._Leaky_Relu(layer, alpha)

        return layer

    def _Residual(self, conv, shortcut, alpha=0.01):
        res = self._Leaky_Relu(conv + shortcut, alpha)

        return res

    def _UpSampling2d(self, inputs, filters, shape=(1, 1), strides=(2, 2), name=None):
        layer = tf.layers.conv2d_transpose(inputs, filters, shape, strides, name=name)

        return layer

    def get_output(self):
        return [self.scale_0, self.scale_1, self.scale_2]


def darknet53(inputs_x, training):
    return DarkNet53(inputs_x, training)