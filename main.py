import numpy

from model import StandardModel
from nmt import parse_args, init_or_restore_variables, read_all_lines
import tensorflow as tf
import util
from zoo import init_nncontext
from zoo.common import Sample

from zoo.pipeline.api.net import TFNet

def preprocess(x, beam_size):
    y_dummy = numpy.zeros(shape=(len(x), 1))
    x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                        maxlen=None)

    x_in = numpy.repeat(x, repeats=2, axis=-1)
    x_mask_in = numpy.repeat(x_mask, repeats=2, axis=-1)
    return [x_in, x_mask_in]


def to_sample(features):
    return Sample.from_ndarray(features, numpy.zeros([features[0].shape[0]]))


if __name__ == "__main__":

    config = parse_args()
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    with tf.Session(config=tf_config) as sess:

        # construct inference model
        model = StandardModel(config)
        beam_size = 2
        beam_ys, parents, cost = model._get_beam_search_outputs(beam_size)
        saver = init_or_restore_variables(config, sess)
        inputs = [model.inputs.raw_x, model.inputs.raw_x_mask]
        outputs = [beam_ys, parents, cost]

        # create TFNet
        tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

        # input data
        sentences = open(config.valid_source_dataset, 'r').readlines()
        batches, idxs = read_all_lines(config, sentences, config.valid_batch_size)
        sc = init_nncontext()
        data_rdd = sc.parallelize(batches[:4]).map(lambda x: preprocess(x, beam_size)).map(lambda x: to_sample(x))

        # prediction
        result_rdd = tfnet.predict(data_rdd, batch_pre_core=1)

        print result_rdd.collect()