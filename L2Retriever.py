
import tensorflow as tf
import numpy as np

class L2Retriever:

    def __init__(self, dim, top_k=3, use_norm=False, use_gpu=True):

        self.dim = dim
        self.top_k = top_k
        self.use_norm = use_norm
        
        if(int(tf.__version__[0]) == 1):
          config = tf.ConfigProto(
              device_count={'GPU': (1 if use_gpu else 0)}
          )
          self.session = tf.Session(config=config)
          self.norm = None
          self.query = tf.placeholder("float", [self.dim])
          self.kbase = tf.placeholder("float", [None, self.dim])

        else:

          config = tf.compat.v1.ConfigProto(
              device_count={'GPU': (1 if use_gpu else 0)}
          )
          self.session = tf.compat.v1.Session(config=config)

          self.norm = None
          tf.compat.v1.disable_eager_execution()
          self.query = tf.compat.v1.placeholder("float", [self.dim])
          self.kbase = tf.compat.v1.placeholder("float", [None, self.dim])
        

        
        self.build_graph()

    def build_graph(self):
      
        if self.use_norm:
          if(int(tf.__version__[0]) >= 2):
            self.norm = tf.compat.v1.placeholder("float", [None, 1])
          else:
            self.norm = tf.placeholder("float", [None, 1])

        distance = dot_l2_distances(self.kbase, self.query, self.norm)
        top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=self.top_k)
        top_dists = tf.negative(top_neg_dists)

        self.top_distances = top_dists
        self.top_indices = top_indices

    def predict(self, kbase, query, norm=None):
        query = np.squeeze(query)
        feed_dict = {self.query: query, self.kbase: kbase}
        if self.use_norm:
          feed_dict[self.norm] = norm
        
        I, D = self.session.run([self.top_indices, self.top_distances],
                                feed_dict=feed_dict)
        
        return I, D


      
def dot_l2_distances(kbase, query, norm=None):
    query = tf.reshape(query, (1, -1))
    
    if norm is None:
      XX = tf.keras.backend.batch_dot(kbase, kbase, axes=1)
    else:
      XX = norm
    YY = tf.matmul(query, tf.transpose(query))
    XY = tf.matmul(kbase, tf.transpose(query))
    
    distance = XX - 2 * XY + YY
    distance = tf.sqrt(tf.reshape(distance, (-1,)))
    
    return distance
