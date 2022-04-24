from typing import List
import numpy as np


class BM25:
    def __init__(self, documents: List, b: float = 0.75, k1: float = 1.2) -> None:
        self.b = b
        self.k1 = k1

        self.num_doc = 0
        self.fieldLens = []
        self.frequency_word_in_doc = []
        self.documents = []

        for document in documents:
            frequency = {}
            document = document.split()
            self.documents.append(document)
            self.num_doc += 1
            self.fieldLens.append(len(document))
            for i in document:
                try:
                    frequency[i] += 1
                except:
                    frequency[i] = 1
            self.frequency_word_in_doc.append(frequency)
        self.avgFieldLen = np.mean(self.fieldLens)

    def cal_idf(self, q: str) -> float:
        f_q = sum([1 if q in i else 0 for i in self.documents])
        return np.log(1+(len(self.documents)-f_q+0.5)/(f_q+0.5))

    def get_score(self, query: str) -> List:
        query = query.split()
        scores = []
        for i, doc in enumerate(self.documents):
            score = 0
            for q in query:
                f = self.frequency_word_in_doc[i].get(q, 0)
                idf = self.cal_idf(q)
                score += idf*((f*(self.k1+1))/(f+self.k1 *
                              (1-self.b+self.b*(len(doc)/self.avgFieldLen))))
            scores.append(score)
        return scores


class BM25Plus(BM25):
    def __init__(self, documents: List, b: float = 0.75, k1: float = 1.2, delta: float = 1.0) -> None:
        super().__init__(documents, b, k1)
        self.delta = delta

    def get_score(self, query: str) -> List:
        query = query.split()
        scores = []
        for i, doc in enumerate(self.documents):
            score = 0
            for q in query:
                f = self.frequency_word_in_doc[i].get(q, 0)
                idf = self.cal_idf(q)
                score += idf*((f*(self.k1+1))/(f+self.k1 *
                              (1-self.b+self.b*(len(doc)/self.avgFieldLen)))+self.delta)
            scores.append(score)
        return scores
