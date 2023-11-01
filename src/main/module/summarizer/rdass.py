from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


class RDASS:
    def __init__(self, sentence_bert: SentenceTransformer = None):
        self.sentence_bert = sentence_bert

    def _cal_cos_sim(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def get_scores(self, docs, ref, predict):
        embedding_docs = self.sentence_bert.encode(docs)
        embedding_ref = self.sentence_bert.encode(ref)
        embedding_predict = self.sentence_bert.encode(predict)

        return self._cal_cos_sim(embedding_predict, embedding_ref) + self._cal_cos_sim(embedding_predict, embedding_docs)
