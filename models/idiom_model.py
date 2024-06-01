import torch
import torch.nn as nn
import torch.optim as optim
from data.idiom_data.idiom_translation import sentence_to_tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden[-1].unsqueeze(0).repeat(src_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=0)

# Define the decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # print("enter decoder forward")
        input = input.unsqueeze(0)  # input shape: [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # embedded shape: [1, batch_size, emb_dim]
        # print("finished embeded")
        # print("hidden shape", hidden.shape)
        # print("encoder_output shape", encoder_outputs.shape)

        a = self.attention(hidden, encoder_outputs)  # a shape: [src_len, batch_size]
        # print("finished attention")
        a = a.unsqueeze(1)  # a shape: [src_len, 1, batch_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs shape: [batch_size, src_len, hid_dim]

        weighted = torch.bmm(a.permute(2, 1, 0), encoder_outputs)  # weighted shape: [batch_size, 1, hid_dim]
        weighted = weighted.permute(1, 0, 2)  # weighted shape: [1, batch_size, hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn_input shape: [1, batch_size, emb_dim + hid_dim]

        output, hidden = self.rnn(rnn_input, hidden)  # output shape: [1, batch_size, hid_dim]
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # prediction shape: [batch_size, output_dim]
        return prediction, hidden

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, vocab, tokenizer):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.vocab = vocab  # Add vocab as a class attribute
        self.tokenizer = tokenizer

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, mode='train'):
        trg_len = trg.shape[0] if trg is not None else 100  # Set a maximum length for inference
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :] if trg is not None else torch.zeros(batch_size, dtype=torch.long).to(self.device)  # Start tokens or zero for inference
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            
            if mode == 'train':
                input = trg[t] if random.random() < teacher_forcing_ratio else top1
            else:
                input = top1

            # If at inference and reach <eos> token, break early
            if mode != 'train' and (input == 2).all():  # Assuming 2 is the <eos> token
                break
        
        return outputs

    def train_model(self, dataloader, optimizer, criterion, clip, vocab):
        self.train()
        epoch_loss = 0

        for i, (src, trg) in enumerate(dataloader):
            src = torch.nn.utils.rnn.pad_sequence(src, padding_value=vocab["<pad>"], batch_first=False).to(self.device)
            trg = torch.nn.utils.rnn.pad_sequence(trg, padding_value=vocab["<pad>"], batch_first=False).to(self.device)

            optimizer.zero_grad()

            output = self(src, trg)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def sample(self, sentence, device=None):
        self.eval()
        tokenizer = self.tokenizer
        tokens = tokenizer(sentence)
        token_indices = [self.vocab[token] for token in tokens]
        if device== None:
            src_tensor = torch.tensor(token_indices).unsqueeze(1).to(self.device)  # Shape: [src_len, 1]
        else:
            print("Enter different device", device)
            src_tensor = torch.tensor(token_indices).unsqueeze(1).to(device)  # Shape: [src_len, 1]
            print(src_tensor.device)


        with torch.no_grad():
            outputs = self.forward(src_tensor, mode='eval')
        
        output_indices = outputs.argmax(-1).squeeze().tolist()
        output_tokens = [self.vocab.get_itos()[index] for index in output_indices if index != self.vocab["<pad>"]]

        # Remove everything after the <eos> token
        if '<eos>' in output_tokens:
            output_tokens = output_tokens[:output_tokens.index('<eos>')]
        
        return ' '.join(output_tokens)

# Example usage
# Assuming model, vocab, and device are already defined
