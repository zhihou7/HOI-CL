import unittest
import _init_paths
import tensorflow as tf
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''
class IOTestCase(tf.test.TestCase):


    def test_Get_Next_Instance_HO_Neg_HICO(self):
        pass

    def test_obtain_data1(self):
        from ult.ult import obtain_data_vcl_hico

        image, img_id, num_pos, Human_augmented, Object_augmented, action_HO, sp = obtain_data_vcl_hico(
            Pos_augment=15, Neg_select=60, augment_type=5, zero_shot_type=0, isalign=True)

        with self.test_session() as sess:
            count_time = 0
            avg_time = 0
            res = sess.run([num_pos, tf.shape(image[0]), tf.shape(image[1]), img_id, num_pos, tf.shape(Human_augmented[0])])
            for i in range(1000):
                import time
                st = time.time()
                res = sess.run([tf.shape(image[0]), tf.shape(image[1]), img_id, num_pos, tf.shape(Human_augmented[0])
                                , tf.shape(Human_augmented[1]), action_HO])
                print(res[3], res[4], res[5], '====')

                avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
                count_time += 1
                print(i, 'generate batch:', time.time() - st, "average;", avg_time)
                st = time.time()

if __name__ == '__main__':
    tf.test.main()
