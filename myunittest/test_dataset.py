import unittest
import _init_paths
import tensorflow as tf
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''
class IOTestCase(tf.test.TestCase):


    def test_Get_Next_Instance_HO_Neg_HICO(self):
        pass


if __name__ == '__main__':
    tf.test.main()
