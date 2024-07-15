import random
from nltk.tokenize import word_tokenize 
import re
import os
from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec

class DatasetENIT:

    def __init__(self):
        self.en_path = "dataset/europarl-v7_en.txt"
        self.it_path = "dataset/europarl-v7_it.txt"
        self.tokenizer_dir = "tokenizer"
        self.tokenizer_en_path = "en_bpe_tokenizer"
        self.tokenizer_it_path = "it_bpe_tokenizer"
        self.word2vec_en_path = "word2vec/word2vec_model_en.model"
        self.word2vec_it_path = "word2vec/word2vec_model_it.model"

    def get_raw_data(self):
        en_list = []
        it_list = []
        with open(self.en_path, 'r', encoding='utf-8') as file:
            en_list = [line.strip() for line in file]

        with open(self.it_path, 'r', encoding='utf-8') as file:
            it_list = [line.strip() for line in file]

        return en_list, it_list


    def get_preprocessed_data(self, sample_p = 0.1):


        en_list, it_list = self.get_raw_data()


        en_list, it_list = self.apply_preprocessing_text(en_list, it_list)

        
        
        if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json")):
            print("training English BPE tokenizer")
            tokenizer_en = self.train_tokenizer(en_list, self.tokenizer_dir, self.tokenizer_en_path)
        else:
            tokenizer_en = ByteLevelBPETokenizer(
                vocab= os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json"),
                merges= os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-merges.txt"),
                add_prefix_space=True, trim_offsets=False
            )

        if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-vocab.json")):
            print("training Italian BPE tokenizer")
            tokenizer_it = self.train_tokenizer(it_list, self.tokenizer_dir, self.tokenizer_it_path)
        else:
            tokenizer_it = ByteLevelBPETokenizer(
                vocab= os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-vocab.json"),
                merges= os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-merges.txt"),
                add_prefix_space=True, trim_offsets=False
            )


        idxs_sampled = random.sample(range(0, len(en_list)), int(len(en_list) * sample_p))
        sampled_en = [en_list[idx] for idx in idxs_sampled]
        sampled_it = [it_list[idx] for idx in idxs_sampled]


        sampled_en =  [["<s>"] + tokenizer_en.encode(doc).tokens + ["</s>"] for doc in sampled_en]
        sampled_it =  [["<s>"] +  tokenizer_it.encode(doc).tokens + ["</s>"] for doc in sampled_it]

    
        return sampled_en, sampled_it
    
    def filter_samples(self, lang_list):
        pattern_xml = r"[<>]"
        idxs = set()
        for idx, doc in enumerate(lang_list):
            if doc.strip() == "":
                continue

            if re.search(pattern_xml, doc):
                continue

            idxs.add(idx)

        return idxs
        


    def apply_preprocessing_text(self, en_list, it_list):
        
        idxs = self.filter_samples(en_list) & self.filter_samples(it_list)
        return [en_list[idx].lower() for idx in sorted(list(idxs))], [it_list[idx].lower() for idx in sorted(list(idxs))]


    def train_tokenizer(self, data, dir, path):
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=False)
        tokenizer.train_from_iterator(data, vocab_size=10000, min_frequency=2, special_tokens=["<s>", "</s>"])
        tokenizer.save_model(dir, path)

        return tokenizer

    def train_word2vec(self):
        en_tokens, it_tokens = self.get_preprocessed_data(sample_p = 1.0)
        print("training English Word2Vec")
        en_model = Word2Vec(
            sentences=en_tokens,
            vector_size=200,
            window=5,
            min_count=1,
            workers=20,
            negative=20,
            epochs=10,
        )
        print("training Italian Word2Vec")
        en_model.save(self.word2vec_en_path)
        it_model = Word2Vec(
            sentences=it_tokens,
            vector_size=200,
            window=5,
            min_count=1,
            workers=20,
            negative=20,
            epochs=10,
        )
        it_model.save(self.word2vec_it_path)

        return en_model, it_model



    def get_vectorized_data(self, sample_p = 0.1):
        en_tokens, it_tokens = self.get_preprocessed_data(sample_p = sample_p)
        if os.path.exists(self.word2vec_en_path) and os.path.exists(self.word2vec_it_path):
            en_model = Word2Vec.load(self.word2vec_en_path).wv
            it_model = Word2Vec.load(self.word2vec_it_path).wv
        else:
            en_model, it_model = self.train_word2vec()


        en_vectors = []
        for tokens in en_tokens:
            en_vectors.append([en_model[token] for token in tokens])

        it_vectors = []
        for tokens in it_tokens:
            it_vectors.append([it_model[token] for token in tokens])


        return en_vectors, it_vectors
    


