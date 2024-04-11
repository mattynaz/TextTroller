import tensorflow as tf
import tensorflow_hub as hub

class SimilarityPredictor():
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embed = hub.load(module_url)

    def run(self, sentences1, sentences2):
        sts_encode1 = tf.nn.l2_normalize(self.embed(sentences1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(sentences2), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
        return sim_scores
