import streamlit as st
from streamlit_utils import read_file, separate_idioms
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title('NLP static idiom translation')
st.write('This is a version of app for the combination of T5 and static dict')


# Add a text input
sent = st.text_input('What do you want to translate: ')
idioms = read_file('../data/idiom_data/total_idioms.txt')

sentences = []
sentences.append(sent)
results = separate_idioms(sentences, idioms)

show_idioms = st.checkbox('Show List of Idioms')

# If checkbox is checked, print out the list of idioms
if show_idioms:
    st.write("List of Idioms:")
    
    for i,idiom in enumerate(idioms):
        st.write(i, idiom)

#T5 debugging
import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys
import os

# # Debugging the environment and paths
# st.write('Python executable:', sys.executable)
# st.write('Python version:', sys.version)
# st.write('sys.path:', sys.path)

# # Check for GPU availability
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     st.write('There are %d GPU(s) available.' % torch.cuda.device_count())
#     st.write('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     st.write('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

# try:
#     # Load the model and tokenizer
#     en_to_vi_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-base")
#     en_to_vi_tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-base")
#     en_to_vi_model.to(device)
#     st.write('Model loaded successfully.')
# except ImportError as e:
#     st.write('ImportError:', str(e))
# except Exception as e:
#     st.write('Error:', str(e))


# Initiate T5 Model
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

device = "cpu"
en_to_vi_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-base")
en_to_vi_tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-base")
en_to_vi_model.to(device)

# Initiate Idiom dict
from models.idiom_dictionary import IdiomTransDict

idioms_file = '../data/idiom_data/total_idioms.txt'
translations_file = '../data/idiom_data/total_translated_idioms.txt'
idiom_dict = IdiomTransDict(idioms_file, translations_file)

# Inference Loop
i =0 
given_parts = results[i]
print(len(given_parts))
if len(given_parts) < 3:
    given_parts = given_parts[1]
    en_to_vi_model.eval()
    tokenized_text = en_to_vi_tokenizer.encode(given_parts, return_tensors="pt").to(device)
    summary_ids = en_to_vi_model.generate(
                    tokenized_text,
                    max_length=128, 
                    num_beams=5,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                )
    output = en_to_vi_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("this is output", output)
else:
    total_sen = []
    idiom_components = given_parts[0]
    non_idiom_componets = given_parts[1]
    sentence_components = given_parts[2]
    plain_en_sentence_components = []
    concatenated_plain_en_sen = ""
    if idiom_components != None:
        for part in sentence_components:
            if part in idiom_components:
                generated_idiom = idiom_dict.get_translation(part)
                print("Idiom part:", part)
                print("Generated idiom:", generated_idiom)
                if "<unk> " in generated_idiom:
                    generated_idiom = generated_idiom[6:]
                plain_en_sentence_components.append(generated_idiom)
                concatenated_plain_en_sen += generated_idiom + " "
            else:
                plain_en_sentence_components.append(part)
                concatenated_plain_en_sen += part + " "
    concatenated_plain_en_sen = concatenated_plain_en_sen[0:-1]
    print("THIS IS PLAIN_EN_SEN_COM:", plain_en_sentence_components)
    print("THIS IS CON_PLAIN_EN_SEN:", concatenated_plain_en_sen)
    tokenized_text = en_to_vi_tokenizer.encode(concatenated_plain_en_sen, return_tensors="pt").to(device)
    summary_ids = en_to_vi_model.generate(
            tokenized_text,
            max_length=128, 
            num_beams=5,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )
    output = en_to_vi_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("This is the translated sentences: ", output)



# Display the input text
if sent:

    st.write(f'This is the result: {output}')




