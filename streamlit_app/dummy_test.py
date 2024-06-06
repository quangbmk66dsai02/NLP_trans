import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

en_to_vi_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-base")
en_to_vi_tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-base")
en_to_vi_model.to(device)
print('done')