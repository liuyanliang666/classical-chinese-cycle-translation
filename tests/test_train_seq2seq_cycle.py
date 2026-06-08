from ccnlp import train_seq2seq


class FakeEmbeddings:
    weight = "embedding-weight"


class FakeSeq2SeqModel:
    def get_input_embeddings(self):
        return FakeEmbeddings()


class FakeDataParallel:
    def __init__(self, module):
        self.module = module


def test_get_input_embedding_weight_unwraps_data_parallel_model():
    wrapped = FakeDataParallel(FakeSeq2SeqModel())

    assert train_seq2seq.get_input_embedding_weight(wrapped) == "embedding-weight"
