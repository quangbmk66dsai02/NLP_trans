import torch

# Convert sentences to tensors
def sentence_to_tensor(sentence, vocab, tokenizer):
    tokens = tokenizer(sentence)
    indexes = [vocab[token] for token in tokens]
    return torch.tensor([vocab["<sos>"]] + indexes + [vocab["<eos>"]])

# Helper function to tokenize and build vocabulary
def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

        
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences, trg_sentences, vocab, tokenizer):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tensor = sentence_to_tensor(self.src_sentences[idx], self.vocab, self.tokenizer)
        trg_tensor = sentence_to_tensor(self.trg_sentences[idx], self.vocab, self.tokenizer)
        return src_tensor, trg_tensor
