class VersionError(Exception):
    pass


def set_seed(new_seed):
    """
    Setts the random seed for:
        numpy
        tensorflow
        os.environ["PYTHONHASHSEED"]
        random
    """
    if isinstance(new_seed, dict):
        raise NotImplementedError("This hasn't been implemented yet.")
    else:
        from numpy.random import seed
        seed(new_seed)

        import tensorflow as tf
        if tf.__version__[:2] == "1.":
            raise VersionError("Tensorflow must be >=2.0.0. YOu have tensorflow "+tf.__version__)
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

def get_seed():
    state = {"numpy": None, "tensorflow": None, "python": None, "random": None}

    import random
    state["random"] = random.getstate()

    from numpy.random import get_state
    state["numpy"] = get_state()

    from tensorflow.random import get_global_generator
    state["tensorflow"] = get_global_generator()

    # We can't get the random state of PYTHONHASHSEED
    # We can using: https://stackoverflow.com/a/41088757/11106801
    # but we can't set it back so no points in calling this function

    raise NotImplementedError("This hasn't been implemented yet.")