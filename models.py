import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the MultiHeadAttention module.

        Args:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q_embed, k_embed, v_embed, lengths, mask=None):
        """
        Perform forward pass through multi-head attention.

        Args:
            q_embed (Tensor): Query embeddings.
            k_embed (Tensor): Key embeddings.
            v_embed (Tensor): Value embeddings.
            lengths (Tensor): Lengths of the sequences (for masking).
            mask (Tensor, optional): Optional mask to apply.

        Returns:
            Tuple[Tensor, Tensor]: Attention output and attention weights.
        """
        batch_size, seq_length_1, embed_dim = q_embed.size()
        batch_size, seq_length_2, embed_dim = k_embed.size()

        # Create a mask to ignore padding tokens
        if mask is None:
            mask = torch.arange(seq_length_2, device=q_embed.device).expand(batch_size, seq_length_2) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)

        # Project queries, keys, and values
        queries = self.q_proj(q_embed).view(batch_size, seq_length_1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.k_proj(k_embed).view(batch_size, seq_length_2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.v_proj(v_embed).view(batch_size, seq_length_2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate the dot products and scale
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # Shape: (batch_size, num_heads, seq_length_1, seq_length_2)

        # Apply the mask
        scores = scores.masked_fill(~mask, float('-inf'))

        # Calculate attention weights
        attention_weights = nn.functional.softmax(scores, dim=-1)  # Shape: (batch_size, num_heads, seq_length_1, seq_length_2)

        # Calculate the context vectors
        context = torch.matmul(attention_weights, values)  # Shape: (batch_size, num_heads, seq_length_1, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_length_1, num_heads, head_dim)
        context = context.view(batch_size, seq_length_1, embed_dim)  # Shape: (batch_size, seq_length_1, embed_dim)

        # Apply the output projection
        output = self.o_proj(context)  # Shape: (batch_size, seq_length_1, embed_dim)

        return output, attention_weights


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the DecoderMultiHeadAttention module.

        Args:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
        """
        super(DecoderMultiHeadAttention, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.enc_dec_attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, encoder_output, src_lengths, tgt_lengths):
        """
        Perform forward pass through the decoder's multi-head attention.

        Args:
            x (Tensor): Target sequence embeddings.
            encoder_output (Tensor): Output from the encoder.
            src_lengths (Tensor): Lengths of source sequences.
            tgt_lengths (Tensor): Lengths of target sequences.

        Returns:
            Tuple[Tensor, Tensor]: Updated target embeddings and attention weights.
        """
        batch_size, tgt_seq_length, embed_dim = x.size()
        _, src_seq_length, _ = encoder_output.size()

        # Create masks
        src_mask = torch.arange(src_seq_length, device=x.device).expand(batch_size, src_seq_length) < src_lengths.unsqueeze(1)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_seq_length)

        tgt_mask = torch.arange(tgt_seq_length, device=x.device).expand(batch_size, tgt_seq_length) < tgt_lengths.unsqueeze(1)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, tgt_seq_length)

        # Causal mask for the decoder to ensure auto-regressive property
        causal_mask = torch.tril(torch.ones(tgt_seq_length, tgt_seq_length, device=x.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, tgt_seq_length, tgt_seq_length)

        # Combine tgt_mask and causal_mask
        combined_tgt_mask = tgt_mask & causal_mask  # Shape: (batch_size, 1, tgt_seq_length, tgt_seq_length)

        # Self-attention
        x_norm = self.layer_norm1(x)
        self_attention_output, self_attention_weights = self.self_attention(x_norm, x_norm, x_norm, tgt_lengths, combined_tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = x + self_attention_output  # Residual connection

        # Encoder-decoder attention
        x_norm = self.layer_norm2(x)
        enc_dec_attention_output, enc_dec_attention_weights = self.enc_dec_attention(
            x_norm, encoder_output, encoder_output, src_lengths, mask=src_mask
        )
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = x + enc_dec_attention_output  # Residual connection

        return x, enc_dec_attention_weights


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout, input_embeded=True, vocab_size=None):
        """
        Initialize the Encoder module.

        Args:
            emb_dim (int): Dimension of the embedding.
            hidden_dim (int): Dimension of the hidden state.
            dropout (float): Dropout probability.
            input_embeded (bool): Whether the input is already embedded.
            vocab_size (int, optional): Size of the vocabulary (if input is not embedded).
        """
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_embeded = input_embeded
        if not input_embeded:
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, dropout=dropout, batch_first=True)

    def forward(self, src):
        """
        Perform forward pass through the encoder.

        Args:
            src (Tensor): Source sequence embeddings.

        Returns:
            Tuple[Tensor, Tensor]: Hidden and cell states of the LSTM.
        """
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout, output_embeded=True):
        """
        Initialize the Decoder module.

        Args:
            output_dim (int): Dimension of the output.
            emb_dim (int): Dimension of the embedding.
            hidden_dim (int): Dimension of the hidden state.
            dropout (float): Dropout probability.
            output_embeded (bool): Whether the output is already embedded.
        """
        super(Decoder, self).__init__()
        self.output_embeded = output_embeded
        self.output_dim = output_dim
        if not output_embeded:
            self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, tgt, hidden, cell):
        """
        Perform forward pass through the decoder.

        Args:
            tgt (Tensor): Target sequence embeddings.
            hidden (Tensor): Hidden state of the LSTM.
            cell (Tensor): Cell state of the LSTM.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Prediction, updated hidden state, and updated cell state.
        """
        output, (hidden, cell) = self.lstm(tgt, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class PositionWiseFC(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.1):
        """
        Initialize the PositionWiseFC module.

        Args:
            emb_dim (int): Dimension of the embedding.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout probability.
        """
        super(PositionWiseFC, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout_att = nn.Dropout(dropout)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, output):
        """
        Perform forward pass through the position-wise feedforward network.

        Args:
            x (Tensor): Input tensor.
            output (Tensor): Output from the attention layer.

        Returns:
            Tensor: Updated tensor after applying position-wise feedforward network.
        """
        x = self.norm1(x + self.dropout(output))
        ff_output = self.fc2(self.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, use_attention=False, n_layers=1, num_heads=4):
        """
        Initialize the Seq2Seq module.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            device (torch.device): Device to run the model on.
            use_attention (bool): Whether to use attention.
            n_layers (int): Number of layers.
            num_heads (int): Number of attention heads.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.use_attention = use_attention
        self.n_layers = n_layers

        if self.use_attention:
            self.attention_layers_encoder = nn.ModuleList([
                MultiHeadAttention(encoder.emb_dim, num_heads) for _ in range(n_layers)
            ])
            self.fcs_encoder = nn.ModuleList([
                PositionWiseFC(encoder.emb_dim, encoder.hidden_dim) for _ in range(n_layers)
            ])
            self.attention_layers_decoder = nn.ModuleList([
                DecoderMultiHeadAttention(encoder.emb_dim, num_heads) for _ in range(n_layers)
            ])
            self.fcs_decoder = nn.ModuleList([
                PositionWiseFC(encoder.emb_dim, encoder.hidden_dim) for _ in range(n_layers)
            ])

    def forward(self, src, tgt, lengths_tgt):
        """
        Perform forward pass through the Seq2Seq model.

        Args:
            src (Tensor): Source sequence tensor.
            tgt (Tensor): Target sequence tensor.
            lengths_tgt (Tensor): Lengths of target sequences.

        Returns:
            Tensor: Output sequence predictions.
        """
        # Embed source sequences if not already embedded
        if not self.encoder.input_embeded:
            embedded_seq = [self.encoder.embedding(seq) for seq in src]
            lengths = torch.tensor([len(seq) for seq in src])
            padded_sequences = pad_sequence(embedded_seq, batch_first=True, padding_value=0)
            src = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

        # Embed target sequences if not already embedded
        if not self.decoder.output_embeded:
            embedded_seq = [self.decoder.embedding(seq) for seq in tgt]
            lengths = torch.tensor([len(seq) for seq in tgt])
            tgt = pad_sequence(embedded_seq, batch_first=True, padding_value=0)

        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        if self.use_attention:
            src, lengths_src = pad_packed_sequence(src, batch_first=True)
            lengths_src = lengths_src.to(self.device)
            lengths_tgt = lengths_tgt.to(self.device)

            for layer_idx in range(self.n_layers):
                output, encoder_attention_weights = self.attention_layers_encoder[layer_idx](src, src, src, lengths_src)
                src = self.fcs_encoder[layer_idx](src, output)

            for layer_idx in range(self.n_layers):
                output, encoder_decoder_attention_weights = self.attention_layers_decoder[layer_idx](tgt, src, lengths_src, lengths_tgt)
                tgt = self.fcs_decoder[layer_idx](tgt, output)

            src = pack_padded_sequence(src, lengths_src.cpu(), batch_first=True, enforce_sorted=False)

        hidden, cell = self.encoder(src)
        input = tgt[:, 0, :]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t, :] = output[:, 0, :]
            input = tgt[:, t, :]

        return outputs
