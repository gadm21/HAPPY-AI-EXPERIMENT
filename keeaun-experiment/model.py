import tensorflow as tf

class Model(object):
    def __init__(self,PATH_TO_CKPT):
        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.detection_graph = None
        self.sess = None
    
    def setup(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

            self.sess = tf.Session(graph=self.detection_graph)