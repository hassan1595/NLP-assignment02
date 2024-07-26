import random
from nltk.tokenize import word_tokenize
import re
import os
from tokenizers import ByteLevelBPETokenizer
from gensim.models import Word2Vec
import pickle


class CharacterTokenizer:
    """
    A class for tokenizing characters.

    Attributes:
        char2id (dict): A dictionary mapping characters to their integer IDs.
        id2char (dict): A dictionary mapping integer IDs to characters.
    """

    def __init__(self, char2id):
        """
        Initialize the CharacterTokenizer.

        Args:
            char2id (dict): A dictionary mapping characters to their integer IDs.
        """
        self.char2id = char2id
        self.id2char = {id: char for char, id in char2id.items()}

    def token_to_id(self, token):
        """
        Convert a token to its corresponding integer ID.

        Args:
            token (str): The token to convert.

        Returns:
            int or None: The integer ID of the token, or None if the token is not found.
        """
        return self.char2id.get(token, None)

    def id_to_token(self, token_id):
        """
        Convert an integer ID to its corresponding token.

        Args:
            token_id (int): The integer ID to convert.

        Returns:
            str or None: The token corresponding to the integer ID, or None if the ID is not found.
        """
        return self.id2char.get(token_id, None)

    def decode(self, token_ids):
        """
        Convert a list of token IDs back to a string.

        Args:
            token_ids (list of int): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        return ''.join([self.id2char[token_id] for token_id in token_ids if token_id in self.id2char])


class DatasetENIT:
    """
    A class to handle dataset processing for English-Italian and English-German translations.

    Attributes:
        en_path (str): Path to the English text file.
        it_path (str): Path to the Italian text file.
        de_2_path (str): Path to the German text file (for DE-EN).
        en_2_path (str): Path to the English text file (for DE-EN).
        tokenizer_dir (str): Directory to save tokenizer models.
        tokenizer_en_path (str): Path for English BPE tokenizer.
        tokenizer_it_path (str): Path for Italian BPE tokenizer.
        tokenizer_de_path (str): Path for German BPE tokenizer.
        tokenizer_character_en_path (str): Path for English character tokenizer.
        tokenizer_character_it_path (str): Path for Italian character tokenizer.
        word2vec_en_path (str): Path for English Word2Vec model.
        word2vec_it_path (str): Path for Italian Word2Vec model.
        word2vec_en_character_path (str): Path for English character-based Word2Vec model.
        word2vec_it_character_path (str): Path for Italian character-based Word2Vec model.
        max_len (int): Maximum sequence length.
        vocab_size_bpe (int): Vocabulary size for BPE tokenization.
    """

    def __init__(self):
        """
        Initialize the DatasetENIT class with default file paths and parameters.
        """
        self.en_path = "dataset/europarl-v7_en.txt"
        self.it_path = "dataset/europarl-v7_it.txt"
        self.de_2_path = "dataset/europarl-v7_2_de.txt"
        self.en_2_path = "dataset/europarl-v7_2_en.txt"
        self.tokenizer_dir = "tokenizer"
        self.tokenizer_en_path = "en_bpe_tokenizer"
        self.tokenizer_it_path = "it_bpe_tokenizer"
        self.tokenizer_de_path = "de_bpe_tokenizer"
        self.tokenizer_character_en_path = "en_character_tokenizer"
        self.tokenizer_character_it_path = "it_character_tokenizer"
        self.word2vec_en_path = "word2vec/word2vec_model_en.model"
        self.word2vec_it_path = "word2vec/word2vec_model_it.model"
        self.word2vec_en_character_path = "word2vec/word2vec_model_en_character.model"
        self.word2vec_it_character_path = "word2vec/word2vec_model_it_character.model"
        self.max_len = 512
        self.vocab_size_bpe = 10000

    def get_raw_data(self, de_en=False):
        """
        Load raw text data from files.

        Args:
            de_en (bool): If True, load DE-EN data. Otherwise, load EN-IT data.

        Returns:
            tuple: A tuple containing two lists of strings: source and target texts.
        """
        if de_en:
            with open(self.de_2_path, 'r', encoding='utf-8') as file:
                de_list = [line.strip() for line in file]

            with open(self.en_2_path, 'r', encoding='utf-8') as file:
                en_list = [line.strip() for line in file]

            return de_list, en_list

        with open(self.en_path, 'r', encoding='utf-8') as file:
            en_list = [line.strip() for line in file]

        with open(self.it_path, 'r', encoding='utf-8') as file:
            it_list = [line.strip() for line in file]

        return en_list, it_list

    def get_preprocessed_data_de_en(self, sample_p=0.1):
        """
        Preprocess data for DE-EN language pairs and train tokenizers if necessary.

        Args:
            sample_p (float): Proportion of data to sample for preprocessing.

        Returns:
            tuple: A tuple containing processed source and target texts and their IDs.
        """
        de_list, en_list = self.get_raw_data(de_en=True)
        de_list, en_list = self.apply_preprocessing_text(de_list, en_list)

        # Train or load BPE tokenizers
        if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_de_path + "-vocab.json")):
            print("Training German BPE tokenizer...")
            tokenizer_de = self.train_tokenizer(de_list, self.tokenizer_dir, self.tokenizer_de_path)
        else:
            tokenizer_de = ByteLevelBPETokenizer(
                vocab=os.path.join(self.tokenizer_dir, self.tokenizer_de_path + "-vocab.json"),
                merges=os.path.join(self.tokenizer_dir, self.tokenizer_de_path + "-merges.txt"),
                add_prefix_space=True, trim_offsets=False
            )

        if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json")):
            print("Training English BPE tokenizer...")
            tokenizer_en = self.train_tokenizer(en_list, self.tokenizer_dir, self.tokenizer_en_path)
        else:
            tokenizer_en = ByteLevelBPETokenizer(
                vocab=os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json"),
                merges=os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-merges.txt"),
                add_prefix_space=True, trim_offsets=False
            )

        self.vocab_size_de = tokenizer_de.get_vocab_size()
        self.vocab_size_en = tokenizer_en.get_vocab_size()

        self.tokenizer_de = tokenizer_de
        self.tokenizer_en = tokenizer_en

        random.seed(42)
        idxs_sampled = random.sample(range(0, len(en_list)), int(len(de_list) * sample_p))
        sampled_de_text = [de_list[idx] for idx in idxs_sampled]
        sampled_en_text = [en_list[idx] for idx in idxs_sampled]

        sampled_de = [["<s>"] + tokenizer_de.encode(doc).tokens[:self.max_len - 2] + ["</s>"] for doc in sampled_de_text]
        sampled_de_ids = [[tokenizer_de.token_to_id("<s>")] + tokenizer_de.encode(doc).ids[:self.max_len - 2] + [tokenizer_de.token_to_id("</s>")] for doc in sampled_de_text]

        sampled_en = [["<s>"] + tokenizer_en.encode(doc).tokens[:self.max_len - 2] + ["</s>"] for doc in sampled_en_text]
        sampled_en_ids = [[tokenizer_en.token_to_id("<s>")] + tokenizer_en.encode(doc).ids[:self.max_len - 2] + [tokenizer_en.token_to_id("</s>")] for doc in sampled_en_text]

        return (sampled_de, sampled_de_ids), (sampled_en, sampled_en_ids)

    def get_preprocessed_data(self, sample_p=0.1, charachter_based=False, de_en=False):
        """
        Preprocess data and train tokenizers for the specified language pairs and tokenization method.

        Args:
            sample_p (float): Proportion of data to sample for preprocessing.
            charachter_based (bool): If True, use character-based tokenization.
            de_en (bool): If True, preprocess DE-EN data. Otherwise, preprocess EN-IT data.

        Returns:
            tuple: A tuple containing processed source and target texts and their IDs.
        """
        if de_en:
            return self.get_preprocessed_data_de_en(sample_p)

        en_list, it_list = self.get_raw_data()
        en_list, it_list = self.apply_preprocessing_text(en_list, it_list)

        # Train or load tokenizers
        if not charachter_based:
            if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json")):
                print("Training English BPE tokenizer...")
                tokenizer_en = self.train_tokenizer(en_list, self.tokenizer_dir, self.tokenizer_en_path)
            else:
                tokenizer_en = ByteLevelBPETokenizer(
                    vocab=os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-vocab.json"),
                    merges=os.path.join(self.tokenizer_dir, self.tokenizer_en_path + "-merges.txt"),
                    add_prefix_space=True, trim_offsets=False
                )

            if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-vocab.json")):
                print("Training Italian BPE tokenizer...")
                tokenizer_it = self.train_tokenizer(it_list, self.tokenizer_dir, self.tokenizer_it_path)
            else:
                tokenizer_it = ByteLevelBPETokenizer(
                    vocab=os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-vocab.json"),
                    merges=os.path.join(self.tokenizer_dir, self.tokenizer_it_path + "-merges.txt"),
                    add_prefix_space=True, trim_offsets=False
                )

            self.vocab_size_en = tokenizer_en.get_vocab_size()
            self.vocab_size_it = tokenizer_it.get_vocab_size()

        else:
            if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_character_en_path)):
                tokenizer_en = self.train_character_tokenizer(en_list, self.tokenizer_dir, self.tokenizer_character_en_path)
            else:
                with open(os.path.join(self.tokenizer_dir, self.tokenizer_character_en_path), 'rb') as file:
                    tokenizer_en = pickle.load(file)

            if not os.path.exists(os.path.join(self.tokenizer_dir, self.tokenizer_character_it_path)):
                tokenizer_it = self.train_character_tokenizer(it_list, self.tokenizer_dir, self.tokenizer_character_it_path)
            else:
                with open(os.path.join(self.tokenizer_dir, self.tokenizer_character_it_path), 'rb') as file:
                    tokenizer_it = pickle.load(file)

            self.vocab_size_en = len(list(tokenizer_en.char2id.items()))
            self.vocab_size_it = len(list(tokenizer_it.char2id.items()))

        self.tokenizer_en = tokenizer_en
        self.tokenizer_it = tokenizer_it

        random.seed(42)
        idxs_sampled = random.sample(range(0, len(en_list)), int(len(en_list) * sample_p))
        sampled_en_text = [en_list[idx] for idx in idxs_sampled]
        sampled_it_text = [it_list[idx] for idx in idxs_sampled]

        if not charachter_based:
            sampled_en = [["<s>"] + tokenizer_en.encode(doc).tokens[:self.max_len - 2] + ["</s>"] for doc in sampled_en_text]
            sampled_en_ids = [[tokenizer_en.token_to_id("<s>")] + tokenizer_en.encode(doc).ids[:self.max_len - 2] + [tokenizer_en.token_to_id("</s>")] for doc in sampled_en_text]

            sampled_it = [["<s>"] + tokenizer_it.encode(doc).tokens[:self.max_len - 2] + ["</s>"] for doc in sampled_it_text]
            sampled_it_ids = [[tokenizer_it.token_to_id("<s>")] + tokenizer_it.encode(doc).ids[:self.max_len - 2] + [tokenizer_it.token_to_id("</s>")] for doc in sampled_it_text]

        else:
            sampled_en = [["<s>"] + list(doc)[:self.max_len - 2] + ["</s>"] for doc in sampled_en_text]
            sampled_en_ids = [[tokenizer_en.token_to_id(token) for token in tokens] for tokens in sampled_en]

            sampled_it = [["<s>"] + list(doc)[:self.max_len - 2] + ["</s>"] for doc in sampled_it_text]
            sampled_it_ids = [[tokenizer_it.token_to_id(token) for token in tokens] for tokens in sampled_it]

        return (sampled_en, sampled_en_ids), (sampled_it, sampled_it_ids)

    def filter_samples(self, lang_list):
        """
        Filter out samples that are empty or contain XML-like tags.

        Args:
            lang_list (list of str): List of documents to filter.

        Returns:
            set: A set of indices of valid documents.
        """
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
        """
        Apply text preprocessing by filtering out invalid samples and converting text to lowercase.

        Args:
            en_list (list of str): List of English texts.
            it_list (list of str): List of Italian texts.

        Returns:
            tuple: A tuple containing two lists: preprocessed English and Italian texts.
        """
        idxs = self.filter_samples(en_list) & self.filter_samples(it_list)
        return [en_list[idx].lower() for idx in sorted(list(idxs))], [it_list[idx].lower() for idx in sorted(list(idxs))]

    def train_tokenizer(self, data, dir, path):
        """
        Train a Byte-Level BPE tokenizer on the given data.

        Args:
            data (list of str): List of texts to train the tokenizer on.
            dir (str): Directory to save the tokenizer model.
            path (str): Path to save the tokenizer model.

        Returns:
            ByteLevelBPETokenizer: The trained tokenizer.
        """
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=False)
        tokenizer.train_from_iterator(data, vocab_size=self.vocab_size_bpe, min_frequency=2, special_tokens=["<s>", "</s>"])
        tokenizer.save_model(dir, path)

        return tokenizer

    def train_character_tokenizer(self, data, dir, path):
        """
        Train a character-based tokenizer on the given data.

        Args:
            data (list of str): List of texts to train the tokenizer on.
            dir (str): Directory to save the tokenizer model.
            path (str): Path to save the tokenizer model.

        Returns:
            CharacterTokenizer: The trained character-based tokenizer.
        """
        vocab = set()
        vocab.update(["<s>", "</s>"])
        for doc in data:
            vocab.update(list(doc))

        char2id = {char: id for id, char in enumerate(sorted(list(vocab)))}
        obj = CharacterTokenizer(char2id)

        with open(os.path.join(dir, path), 'wb') as file:
            pickle.dump(obj, file)

        return obj

    def train_word2vec(self, eng_path, it_path, charachter_based):
        """
        Train Word2Vec models for English and Italian languages.

        Args:
            eng_path (str): Path to save the English Word2Vec model.
            it_path (str): Path to save the Italian Word2Vec model.
            charachter_based (bool): If True, use character-based tokenization.

        Returns:
            tuple: A tuple containing the Word2Vec models for English and Italian.
        """
        (en_tokens, en_ids), (it_tokens, it_ids) = self.get_preprocessed_data(sample_p=1.0, charachter_based=charachter_based)
        
        print("Training English Word2Vec...")
        en_model = Word2Vec(
            sentences=en_tokens,
            vector_size=200,
            window=5,
            min_count=1,
            workers=20,
            negative=50,
            epochs=10,
        )
        en_model.save(eng_path)
        
        print("Training Italian Word2Vec...")
        it_model = Word2Vec(
            sentences=it_tokens,
            vector_size=200,
            window=5,
            min_count=1,
            workers=20,
            negative=50,
            epochs=10,
        )
        it_model.save(it_path)

        return en_model.wv, it_model.wv

    def get_vectorized_data(self, sample_p=0.1, charachter_based=False, de_en=False):
        """
        Get vectorized data using Word2Vec models.

        Args:
            sample_p (float): Proportion of data to sample for vectorization.
            charachter_based (bool): If True, use character-based tokenization.
            de_en (bool): If True, process DE-EN data. Otherwise, process EN-IT data.

        Returns:
            tuple: A tuple containing the vectorized source and target texts and their IDs.
        """
        if de_en:
            print("German-English vectorization is not supported. IDs only are returned.")
            (de_tokens, de_ids), (en_tokens, en_ids) = self.get_preprocessed_data(sample_p=sample_p, charachter_based=charachter_based, de_en=True)
            return (None, de_ids), (None, en_ids)

        (en_tokens, en_ids), (it_tokens, it_ids) = self.get_preprocessed_data(sample_p=sample_p, charachter_based=charachter_based)

        # Load or train Word2Vec models
        if not charachter_based:
            if os.path.exists(self.word2vec_en_path) and os.path.exists(self.word2vec_it_path):
                en_model = Word2Vec.load(self.word2vec_en_path).wv
                it_model = Word2Vec.load(self.word2vec_it_path).wv
            else:
                en_model, it_model = self.train_word2vec(self.word2vec_en_path, self.word2vec_it_path, charachter_based)
        else:
            if os.path.exists(self.word2vec_en_character_path) and os.path.exists(self.word2vec_it_character_path):
                en_model = Word2Vec.load(self.word2vec_en_character_path).wv
                it_model = Word2Vec.load(self.word2vec_it_character_path).wv
            else:
                en_model, it_model = self.train_word2vec(self.word2vec_en_character_path, self.word2vec_it_character_path, charachter_based)

        # Convert tokens to vectors
        en_vectors = [[en_model[token] for token in tokens] for tokens in en_tokens]
        it_vectors = [[it_model[token] for token in tokens] for tokens in it_tokens]

        self.en_wv = en_model
        self.it_wv = it_model

        return (en_vectors, en_ids), (it_vectors, it_ids)
