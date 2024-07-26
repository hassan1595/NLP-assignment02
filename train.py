import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from models import *
from tqdm import tqdm
import logging
from dataset import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import shutil
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class TrainRNN:
    def __init__(self, log_path="logs/logs.txt", input_embeded = False, lr = 1e-3, batch_size = 64, n_epochs = 10, en_it = True, de_en = False, save_path = "rnn_models/lstm_en_it_embeded.pt", sample_p = 0.1, charachter_based = False, use_attention = False, attention_plot_path = "plots/attention_plots"):
        """
        Initialize the training class with the given parameters.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.test_batch_size = 130
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_path = save_path
        self.en_it = en_it
        self.de_en = de_en
        self.use_attention = use_attention
        self.d = DatasetENIT()
        self.input_embeded = input_embeded
        self.max_gen_length = self.d.max_len
        self.attention_plot_path = attention_plot_path
        self.start_token = "<s>"
        self.end_token = "</s>"
    
        if de_en and (charachter_based or input_embeded):
            raise ValueError("DE_EN Translation and character_based or input_embeded together is not supported")

        if de_en:
            (_, de_ids), (_, en_ids)= self.d.get_vectorized_data(sample_p= sample_p, charachter_based = charachter_based, de_en=True)
            self.de_ids_train,  self.de_ids_test, self.en_ids_train,  self.en_ids_test  = train_test_split(de_ids,  en_ids, test_size=0.2, random_state=42)

        else:
            (en_vectors, en_ids), (it_vectors, it_ids)= self.d.get_vectorized_data(sample_p= sample_p, charachter_based = charachter_based)
            self.en_vectors_train,  self.en_vectors_test, self.en_ids_train,  self.en_ids_test, self.it_vectors_train,  self.it_vectors_test, self.it_ids_train,  self.it_ids_test = train_test_split(en_vectors, en_ids, it_vectors,  it_ids, test_size=0.2, random_state=42)

        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("\n%(message)s\n")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if de_en:
            vocab_size_src = self.d.vocab_size_de
            vocab_size_tgt = self.d.vocab_size_en
        else:
            if en_it:
                vocab_size_src = self.d.vocab_size_en
                vocab_size_tgt = self.d.vocab_size_it
            else:
                vocab_size_src = self.d.vocab_size_it
                vocab_size_tgt = self.d.vocab_size_en
        encoder = Encoder(emb_dim=200, hidden_dim = 500, dropout = 0.2, input_embeded = input_embeded, vocab_size= vocab_size_src).to(self.device)
        decoder = Decoder(output_dim=vocab_size_tgt, emb_dim=200, hidden_dim = 500,  dropout = 0.2, output_embeded = input_embeded)
        self.seq2seq = Seq2Seq(encoder, decoder, self.device, use_attention).to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def check_and_create_folder(self, folder_path):
        """
        Check if a folder exists, delete it if it does, and create a new one.

        Args:
        folder_path (str): The path of the folder to check, delete, and create.
        """
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' deleted.")
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

    def train(self):
        """
        Train the RNN model.
        """
        optimizer = torch.optim.Adam(
            self.seq2seq.parameters(), lr=self.lr
        )
        self.seq2seq.train()
        for epoch_idx in range(1, self.n_epochs + 1):
            loss_epoch = 0
            total_tokens = 0
            for batch_idx in tqdm(range(0, len(self.en_ids_train), self.batch_size), f"Training Epoch {epoch_idx}"):
                
                optimizer.zero_grad()

                if self.input_embeded:

                    if self.en_it :
                        inp = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.en_vectors_train[batch_idx :batch_idx + self.batch_size]]
                        lengths_inp = torch.tensor([len(seq) for seq in inp])
                        padded_sequences_inp = pad_sequence(inp, batch_first=True, padding_value=0)
                        packed_input_inp = pack_padded_sequence(padded_sequences_inp, lengths_inp, batch_first=True, enforce_sorted=False)
                    
                        out = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.it_vectors_train[batch_idx :batch_idx + self.batch_size]]
                        lengths_out = torch.tensor([len(seq) for seq in out])
                        padded_sequences_out = pad_sequence(out, batch_first=True, padding_value=0)

                        targets = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_train[batch_idx :batch_idx + self.batch_size] ]
                    else:
                        inp = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.it_vectors_train[batch_idx :batch_idx + self.batch_size]]
                        lengths_inp = torch.tensor([len(seq) for seq in inp])
                        padded_sequences_inp = pad_sequence(inp, batch_first=True, padding_value=0)
                        packed_input_inp = pack_padded_sequence(padded_sequences_inp, lengths_inp, batch_first=True, enforce_sorted=False)
                    
                        out = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.en_vectors_train[batch_idx :batch_idx + self.batch_size]]
                        lengths_out = torch.tensor([len(seq) for seq in out])
                        padded_sequences_out = pad_sequence(out, batch_first=True, padding_value=0)

                        targets = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_train[batch_idx :batch_idx + self.batch_size] ]

                    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
                    lstm_output_flat = self.seq2seq(packed_input_inp, padded_sequences_out, lengths_out)
                   
                else:
                    
                    if self.de_en:
                        inp = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.de_ids_train[batch_idx :batch_idx + self.batch_size] ]
                        out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_train[batch_idx :batch_idx + self.batch_size] ]
                        
                    else:
                        if self.en_it:
                            inp = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_train[batch_idx :batch_idx + self.batch_size] ]
                            out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_train[batch_idx :batch_idx + self.batch_size] ]
                        else:
                            inp = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_train[batch_idx :batch_idx + self.batch_size] ]
                            out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_train[batch_idx :batch_idx + self.batch_size] ]

                    lengths_out = torch.tensor([len(seq) for seq in out])
                    targets = out
                    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
                    lstm_output_flat = self.seq2seq(inp, out, lengths_out)


                max_len_seqs = lstm_output_flat.shape[1]
                lstm_output_flat = lstm_output_flat.view(-1, self.d.vocab_size_it if self.en_it else self.d.vocab_size_en)

                targets_flat = padded_targets.view(-1).to(self.device)

                mask = torch.arange(max_len_seqs).expand(len(lengths_out), max_len_seqs) < lengths_out.unsqueeze(1)
                mask_flat = mask.view(-1)

                lstm_output_flat_masked = lstm_output_flat[mask_flat]
                targets_flat_masked = targets_flat[mask_flat]

                loss = self.criterion(lstm_output_flat_masked, targets_flat_masked) 
                loss.backward()
                optimizer.step()
                loss_epoch += len(targets_flat_masked) * loss.item()
                total_tokens += len(targets_flat_masked)
        
            print(f"Epoch {epoch_idx} - Loss {loss_epoch/total_tokens}")

        torch.save(self.seq2seq.state_dict(), self.save_path)


    def plot_attention_heatmap(self, attention_weights, src_tokens, tgt_tokens, save_path=None):
        """
        Plot an attention heat map with the source language tokens on the horizontal axis
        and the target language tokens on the vertical axis.

        Args:
        attention_weights (numpy.ndarray): The attention weights matrix of size (target_len, source_len).
        src_tokens (list of str): The source language tokens.
        tgt_tokens (list of str): The target language tokens.
        save_path (str, optional): The file path to save the plot. If None, the plot will be shown but not saved.
        """
        # Check that the attention weights matrix matches the lengths of the token lists
        assert attention_weights.shape == (len(tgt_tokens), len(src_tokens)), "Shape of attention_weights must match length of tgt_tokens and src_tokens."

        fig, ax = plt.subplots()

        # Plot the heatmap
        cax = ax.matshow(attention_weights, cmap='viridis')

        # Set up axes
        ax.set_xticks(np.arange(len(src_tokens)))
        ax.set_yticks(np.arange(len(tgt_tokens)))

        # Label each axis with the respective tokens
        ax.set_xticklabels( src_tokens, rotation=90)
        ax.set_yticklabels(tgt_tokens)

        # Show color bar
        fig.colorbar(cax)

        # Set labels for the axes
        ax.set_xlabel('Source Language Tokens')
        ax.set_ylabel('Target Language Tokens')

        # Save or show the plot
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
    
        
    def generate_sequence(self, hidden, cell, src = None, lenghts_src= None, return_attention = False):
        """
        Generates a sequence based on the given hidden and cell states
        """
        if self.input_embeded:
            if self.en_it:
                inputs = torch.tensor([[self.d.it_wv[self.start_token]]], dtype = torch.float32, device=self.device)
            else:
                inputs = torch.tensor([[self.d.en_wv[self.start_token]]], dtype = torch.float32, device=self.device)
        else:
            if self.de_en:
                inputs = self.seq2seq.decoder.embedding(torch.tensor([[self.d.tokenizer_en.token_to_id(self.start_token)]], dtype = torch.int64, device=self.device))
            else:

                if self.en_it:
                    inputs = self.seq2seq.decoder.embedding(torch.tensor([[self.d.tokenizer_it.token_to_id(self.start_token)]], dtype = torch.int64, device=self.device))
                else:
                    inputs = self.seq2seq.decoder.embedding(torch.tensor([[self.d.tokenizer_en.token_to_id(self.start_token)]], dtype = torch.int64, device=self.device))
       
        if self.de_en:
            generated_sequence = [self.d.tokenizer_en.token_to_id(self.start_token)]
        else:

            if self.en_it:
                generated_sequence = [self.d.tokenizer_it.token_to_id(self.start_token)]

            else:
                generated_sequence = [self.d.tokenizer_en.token_to_id(self.start_token)]

        for gen_idx in range(self.max_gen_length - 2):
            output, hidden_decoder, cell_decoder = self.seq2seq.decoder(inputs, hidden, cell)
            topv, topi = output.topk(1)
            next_token_id = topi[-1].item()
            if self.en_it:
                next_token = self.d.tokenizer_it.id_to_token(next_token_id)
            else:
                next_token = self.d.tokenizer_en.id_to_token(next_token_id)
            generated_sequence.append(next_token_id)

            if next_token == self.end_token:
                break

            if self.input_embeded:
                if self.en_it:
                    inputs = torch.cat([inputs, torch.tensor([[self.d.it_wv[next_token]]], dtype = torch.float32, device=self.device)], dim = 1)
                else:
                    inputs = torch.cat([inputs, torch.tensor([[self.d.en_wv[next_token]]], dtype = torch.float32, device=self.device)], dim = 1)
            else:
                if self.en_it:
                    inputs = torch.cat([inputs, self.seq2seq.decoder.embedding(torch.tensor([[self.d.tokenizer_it.token_to_id(next_token)]], dtype = torch.int64, device=self.device))], dim = 1)
                else:
                    inputs = torch.cat([inputs, self.seq2seq.decoder.embedding(torch.tensor([[self.d.tokenizer_en.token_to_id(next_token)]], dtype = torch.int64, device=self.device))], dim = 1)

            if self.use_attention:
                for layer_idx in range(self.seq2seq.n_layers):
                    lenghts_tgt = torch.tensor([len(inp) for inp in inputs]).to(self.device)
                    output, encoder_decoder_attention_weights = self.seq2seq.arrention_layers_decoder[layer_idx](inputs, src, lenghts_src, lenghts_tgt)
                    inputs = self.seq2seq.fcs_decoder[layer_idx](inputs, output)

        if return_attention and self.use_attention:
            return generated_sequence, encoder_decoder_attention_weights
        return generated_sequence

    def test(self):
        """
         Evaluate the trained model on the test set and compute BLEU score and Perplexity.
        """
        if self.de_en:
            raise ValueError("Please use test_pivot instead")
        
        with torch.no_grad():
            self.seq2seq.load_state_dict(torch.load(self.save_path))
            self.seq2seq.eval()

            bleus = []
            perps = []
            seq_lens = []
            if self.use_attention:
                attention_plot_idx = 0
                self.check_and_create_folder(self.attention_plot_path)
            self.logger.info("Translation Samples: \n")
            for batch_idx in tqdm(range(0, len(self.en_ids_test), self.test_batch_size), "Testing"):
                if self.input_embeded:

                    if self.en_it:
                        inp = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.en_vectors_test[batch_idx :batch_idx + self.test_batch_size]]
                        lenghts_src = torch.tensor([len(seq) for seq in inp])
                        out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        lenghts_tgt = torch.tensor([len(seq) for seq in out])
                        src = pad_sequence(inp, batch_first=True, padding_value=0)
                        src_packed = pack_padded_sequence(src, lenghts_src, batch_first=True, enforce_sorted=False)

                        src_aval = src_packed
                        out_v = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.it_vectors_test[batch_idx :batch_idx + self.test_batch_size]]
                        tgt_eval = pad_sequence(out_v, batch_first=True, padding_value=0)

                        
                    else:
                        inp = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.it_vectors_test[batch_idx :batch_idx + self.test_batch_size]]
                        lenghts_src = torch.tensor([len(seq) for seq in inp])
                        out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        lenghts_tgt = torch.tensor([len(seq) for seq in out])
                        src = pad_sequence(inp, batch_first=True, padding_value=0)
                        src_packed = pack_padded_sequence(src, lenghts_src, batch_first=True, enforce_sorted=False)

                        src_aval = src_packed
                        out_v = [ torch.tensor(np.array(v), dtype = torch.float32).to(self.device) for v in self.en_vectors_test[batch_idx :batch_idx + self.test_batch_size]]
                        tgt_eval = pad_sequence(out_v, batch_first=True, padding_value=0)

                else:
                    if self.en_it:
                        inp = [ self.seq2seq.encoder.embedding(torch.tensor(np.array(ids), dtype = torch.int64).to(self.device)) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size]]
                        lenghts_src = torch.tensor([len(seq) for seq in inp])
                        out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        lenghts_tgt = torch.tensor([len(seq) for seq in out])
                        src = pad_sequence(inp, batch_first=True, padding_value=0)
                        src_packed = pack_padded_sequence(src, lenghts_src, batch_first=True, enforce_sorted=False)


                        src_aval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        tgt_eval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_test[batch_idx :batch_idx + self.test_batch_size] ]



                    else:

                        inp = [ self.seq2seq.encoder.embedding(torch.tensor(np.array(ids), dtype = torch.int64).to(self.device)) for ids in self.it_ids_test[batch_idx :batch_idx + self.test_batch_size]]
                        lenghts_src = torch.tensor([len(seq) for seq in inp])
                        out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        lenghts_tgt = torch.tensor([len(seq) for seq in out])
                        src = pad_sequence(inp, batch_first=True, padding_value=0)
                        src_packed = pack_padded_sequence(src, lenghts_src, batch_first=True, enforce_sorted=False)

                        src_aval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.it_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                        tgt_eval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
            
                if self.use_attention:
                    src, lenghts_src = pad_packed_sequence(src_packed, batch_first=True) 
                    lenghts_src  = lenghts_src.to(self.device)
                    lenghts_tgt  = lenghts_tgt.to(self.device)
                    for layer_idx in range(self.seq2seq.n_layers):
                        output, encoder_attention_weights = self.seq2seq.arrention_layers_encoder[layer_idx](src, src, src, lenghts_src)
                        src = self.seq2seq.fcs_encoder[layer_idx](src, output)
                    src_packed = pack_padded_sequence(src, lenghts_src.cpu(), batch_first=True, enforce_sorted=False)

                hidden, cell = self.seq2seq.encoder(src_packed)


         
                lstm_output = self.seq2seq(src_aval, tgt_eval, lenghts_tgt)
     

                for i in range(len(inp)):
          
                    perp = (torch.tensor([ nn.Softmax(dim=-1)(preds)[out[i][j]] ** (-1/len(out[i])) for j, preds in enumerate(lstm_output[i][:lenghts_tgt[i]] )])[1:-1] ).prod() 
                    perps.append(perp)
                    seq_lens.append(len(inp[i]))
                    if self.use_attention and attention_plot_idx < 20 and len(out[i]) <= 20:
                        generated_sequence, attention_weights = self.generate_sequence(hidden[:, i:i+1, :], cell[:, i:i+1, :], src = src[i:i+1, :, :], lenghts_src = lenghts_src[i:i+1], return_attention=True)
                        # Taking the mean over all heads execluding the start and the end token
                        attention_weights = attention_weights.squeeze().mean(dim = 0)[1:, 1:len(inp[i]) -1]
                        if self.en_it:
                            src_tokens = [ self.d.tokenizer_en.id_to_token(id) for id in self.en_ids_test[batch_idx + i]][1:-1]
                            src_tokens = [token[1:] if token[0] == "Ġ" else token  for token in src_tokens]
                            tgt_tokens = [ self.d.tokenizer_it.id_to_token(id) for id in generated_sequence][1:-1]
                            tgt_tokens = [token[1:] if token[0] == "Ġ" else token  for token in tgt_tokens]
                        else:
                            src_tokens = [ self.d.tokenizer_it.id_to_token(id) for id in self.it_ids_test[batch_idx + i]][1:-1]
                            src_tokens = [token[1:] if token[0] == "Ġ" else token  for token in src_tokens]
                            tgt_tokens = [ self.d.tokenizer_en.id_to_token(id) for id in generated_sequence][1:-1]
                            tgt_tokens = [token[1:] if token[0] == "Ġ" else token  for token in tgt_tokens]

                        self.plot_attention_heatmap(attention_weights.cpu().numpy(), src_tokens, tgt_tokens, os.path.join(self.attention_plot_path, f"attenion_plot_{attention_plot_idx}.png"))
            
                        attention_plot_idx += 1

                    generated_sequence = self.generate_sequence(hidden[:, i:i+1, :], cell[:, i:i+1, :], src = src[i:i+1, :, :], lenghts_src = lenghts_src[i:i+1] )
                    
                    if batch_idx < 5:
                        if self.en_it:
                            self.logger.info("\n\nInput: ")
                            self.logger.info(self.d.tokenizer_en.decode(self.en_ids_test[batch_idx + i][1:-1]))
                            self.logger.info("\nOutput: ")
                            self.logger.info(self.d.tokenizer_it.decode(generated_sequence[1:-1]))
                        else:
                            self.logger.info("\n\nInput: ")
                            self.logger.info(self.d.tokenizer_it.decode(self.it_ids_test[batch_idx + i][1:-1]))
                            self.logger.info("\nOutput: ")
                            self.logger.info(self.d.tokenizer_en.decode(generated_sequence[1:-1]))

                    smoothing_function = SmoothingFunction().method1
                    score = sentence_bleu([out[i].cpu().numpy().tolist()[1:-1]], generated_sequence[1:-1], smoothing_function=smoothing_function)
                    bleus.append(score)
                
            bleus_arr = np.array(bleus)
            perps_arr = np.array(perps)
            lens_arr = np.array(seq_lens)
            len_mean = lens_arr.mean()

            self.logger.info("\n\nBLEU: ")
            bleu_mean = bleus_arr.mean()
            self.logger.info(bleu_mean)
            self.logger.info("\n\nPerplexity: ")
            perp_mean = perps_arr.mean()
            self.logger.info(perp_mean)

            self.logger.info("\n\BLEU - Length Covariance: ")
            self.logger.info(((bleus_arr - bleu_mean) * (lens_arr - len_mean)).mean())
            self.logger.info("\n\nPerplexity - Length Covariance: ")
            self.logger.info(((perps_arr - perp_mean) * (lens_arr - len_mean)).mean())


    def test_pivot(self, en_it_pbj):
        """
        Evaluate the Model on German-Italian Translation using English as Pivot Language
        """
        if not  self.de_en or self.input_embeded or en_it_pbj.input_embeded:
            raise ValueError("Configuration not supported")
        
        with torch.no_grad():
            self.seq2seq.load_state_dict(torch.load(self.save_path))
            self.seq2seq.eval()
            en_it_pbj.seq2seq.load_state_dict(torch.load(en_it_pbj.save_path))
            en_it_pbj.seq2seq.eval()

            bleus = []
            perps = []
            seq_lens = []
            for batch_idx in tqdm(range(0, len(self.en_ids_test), self.test_batch_size), "Testing"):
                
                inp = [ self.seq2seq.encoder.embedding(torch.tensor(np.array(ids), dtype = torch.int64).to(self.device)) for ids in self.de_ids_test[batch_idx :batch_idx + self.test_batch_size]]
                lenghts_src = torch.tensor([len(seq) for seq in inp])
                out = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                lenghts_tgt = torch.tensor([len(seq) for seq in out])
                src = pad_sequence(inp, batch_first=True, padding_value=0)
                src_packed = pack_padded_sequence(src, lenghts_src, batch_first=True, enforce_sorted=False)


                src_aval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.de_ids_test[batch_idx :batch_idx + self.test_batch_size] ]
                tgt_eval = [ torch.tensor(np.array(ids, dtype = np.int64)).to(self.device) for ids in self.en_ids_test[batch_idx :batch_idx + self.test_batch_size] ]

                if self.use_attention:
                    src, lenghts_src = pad_packed_sequence(src_packed, batch_first=True) 
                    lenghts_src  = lenghts_src.to(self.device)
                    lenghts_tgt  = lenghts_tgt.to(self.device)
                    for layer_idx in range(self.seq2seq.n_layers):
                        output, encoder_attention_weights = self.seq2seq.arrention_layers_encoder[layer_idx](src, src, src, lenghts_src)
                        src = self.seq2seq.fcs_encoder[layer_idx](src, output)
                    src_packed = pack_padded_sequence(src, lenghts_src.cpu(), batch_first=True, enforce_sorted=False)

                hidden, cell = self.seq2seq.encoder(src_packed)
                lstm_output = self.seq2seq(src_aval, tgt_eval, lenghts_tgt)

                new_inp = []
                generated_texts = []
                for i in range(len(inp)):
          
                    perp = (torch.tensor([ nn.Softmax(dim=-1)(preds)[out[i][j]] ** (-1/len(out[i])) for j, preds in enumerate(lstm_output[i][:lenghts_tgt[i]] )])[1:-1] ).prod() 
                    perps.append(perp)
                    seq_lens.append(len(inp[i]))

                    generated_sequence = self.generate_sequence(hidden[:, i:i+1, :], cell[:, i:i+1, :], src = src[i:i+1, :, :], lenghts_src = lenghts_src[i:i+1] )
                    
                    
                    if batch_idx < 5:
                            new_inp.append(en_it_pbj.seq2seq.encoder.embedding(torch.tensor(np.array(generated_sequence, dtype = np.int64)).to(en_it_pbj.device) ))
                            generated_texts.append((self.d.tokenizer_de.decode(self.de_ids_test[batch_idx + i][1:-1]), self.d.tokenizer_en.decode(generated_sequence[1:-1])))
                    

                    smoothing_function = SmoothingFunction().method1
                    score = sentence_bleu([out[i].cpu().numpy().tolist()[1:-1]], generated_sequence[1:-1], smoothing_function=smoothing_function)
                    bleus.append(score)

                if batch_idx < 5:
                    new_lenghts_src = torch.tensor([len(seq) for seq in new_inp]).to(en_it_pbj.device)
                    new_src = pad_sequence(new_inp, batch_first=True, padding_value=0)
                    new_src_packed = pack_padded_sequence(new_src, new_lenghts_src.cpu(), batch_first=True, enforce_sorted=False)
                    if en_it_pbj.use_attention:
                        new_src, new_lenghts_src = pad_packed_sequence(new_src_packed, batch_first=True) 
                        new_lenghts_src  = new_lenghts_src.to(self.device)
                        for layer_idx in range(en_it_pbj.seq2seq.n_layers):
                            output, encoder_attention_weights = en_it_pbj.seq2seq.arrention_layers_encoder[layer_idx](new_src, new_src, new_src, new_lenghts_src)
                            new_src = en_it_pbj.seq2seq.fcs_encoder[layer_idx](new_src, output)
                        new_src_packed = pack_padded_sequence(new_src, new_lenghts_src.cpu(), batch_first=True, enforce_sorted=False)

                    new_hidden, new_cell = en_it_pbj.seq2seq.encoder(new_src_packed)

                    for i in range(len(new_inp)):
                        new_generated_sequence = en_it_pbj.generate_sequence(new_hidden[:, i:i+1, :], new_cell[:, i:i+1, :], src = new_src[i:i+1, :, :], lenghts_src = new_lenghts_src[i:i+1])
                        self.logger.info("\n\nGerman: ")
                        self.logger.info(generated_texts[i][0])
                        self.logger.info("\n\nEnglish-Pivot: ")
                        self.logger.info(generated_texts[i][1])
                        self.logger.info("\n\nItalian: ")
                        self.logger.info(en_it_pbj.d.tokenizer_it.decode(new_generated_sequence[1:-1]))
    


            self.logger.info("\n\n German-English Translation Metrics:")
            bleus_arr = np.array(bleus)
            perps_arr = np.array(perps)
            lens_arr = np.array(seq_lens)
            len_mean = lens_arr.mean()

            self.logger.info("\n\nBLEU: ")
            bleu_mean = bleus_arr.mean()
            self.logger.info(bleu_mean)
            self.logger.info("\n\nPerplexity: ")
            perp_mean = perps_arr.mean()
            self.logger.info(perp_mean)

            self.logger.info("\n\BLEU - Length Covariance: ")
            self.logger.info(((bleus_arr - bleu_mean) * (lens_arr - len_mean)).mean())
            self.logger.info("\n\nPerplexity - Length Covariance: ")
            self.logger.info(((perps_arr - perp_mean) * (lens_arr - len_mean)).mean())
                  
def main():
    """
    Train and Evaluate all Configurations
    """
    t = TrainRNN(log_path = "logs/lstm_en_it_fixed_embedding.txt", input_embeded = True, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_fixed_embedding.pt", charachter_based = False, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_en_it_learned_embedding.txt", input_embeded = False, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_learned_embedding.pt", charachter_based = False, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_it_en_fixed_embedding.txt", input_embeded = True, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_fixed_embedding.pt", charachter_based = False, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_it_en_learned_embedding.txt", input_embeded = False, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_learned_embedding.pt", charachter_based = False, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_en_it_charachter_fixed_embedding.txt", input_embeded = True, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_charachter_fixed_embedding.pt", charachter_based = True, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_en_it_charachter_learned_embedding.txt", input_embeded = False, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_charachter_learned_embedding.pt", charachter_based = True, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_it_en_charachter_fixed_embedding.txt", input_embeded = True, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_charachter_fixed_embedding.pt", charachter_based = True, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_it_en_charachter_learned_embedding.txt", input_embeded = False, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_charachter_learned_embedding.pt", charachter_based = True, use_attention=False)
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_en_it_fixed_embedding_attention.txt", input_embeded = True, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_fixed_embedding_attention.pt", charachter_based = False, use_attention=True, attention_plot_path = "plots/lstm_en_it_fixed_embedding_attention")
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_en_it_learned_embedding_attention.txt", input_embeded = False, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_learned_embedding_attention.pt", charachter_based = False, use_attention=True, attention_plot_path = "plots/lstm_en_it_learned_embedding_attention")
    t.train()
    t.test()
    

    t = TrainRNN(log_path = "logs/lstm_it_en_fixed_embedding_attention.txt", input_embeded = True, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_fixed_embedding_attention.pt", charachter_based = False, use_attention=True, attention_plot_path = "plots/lstm_it_en_fixed_embedding_attention")
    t.train()
    t.test()

    t = TrainRNN(log_path = "logs/lstm_it_en_learned_embedding_attention.txt", input_embeded = False, en_it= False, de_en= False, save_path= "rnn_models/lstm_it_en_learned_embedding_attention.pt", charachter_based = False, use_attention=True, attention_plot_path = "plots/lstm_it_en_learned_embedding_attention")
    t.train()
    t.test()

    t = TrainRNN(log_path = "logs/lstm_de_en_it_learned_embedding_attention.txt", input_embeded = False, en_it= False, de_en= True, save_path= "rnn_models/lstm_de_en_learned_embedding_attention.pt", charachter_based = False, use_attention=True)
    t.train()
    t.test_pivot(en_it_pbj = TrainRNN(log_path = "logs/lstm_en_it_learned_embedding_attention.txt", input_embeded = False, en_it= True, de_en= False, save_path= "rnn_models/lstm_en_it_learned_embedding_attention.pt", charachter_based = False, use_attention=True, attention_plot_path = "plots/lstm_en_it_learned_embedding_attention"))
if __name__ == "__main__":
    main()

