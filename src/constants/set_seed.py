def set_seed(new_seed=None, pyseed=None, randomseed=None,
             numpyseed=None, tensorflowseed=None):
    """
    Setts the random seed for:
        numpy
        tensorflow
        os.environ["PYTHONHASHSEED"]
        random
    """
    from numpy.random import seed
    seed(new_seed)

    import tensorflow as tf
    tf.random.set_seed(new_seed)

    import os
    os.environ["PYTHONHASHSEED"]=str(new_seed)

    import random
    random.seed(new_seed)

    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
