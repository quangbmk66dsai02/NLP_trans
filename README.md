# NLP Translation Project

This is the repository for NLP project focusing on Static Idiom Translation.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install torch and transformers modules. 

Please note that the the installation of torch should match with your current CUDA installation to prevent any troubles. For more details of previous version compatibility, please visit [torch vision](https://pytorch.org/get-started/previous-versions/).

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers 
```

## Usage
We offer 2 configurations: the combination of T5 with a dictionary (less technical) and the, it can be run in the combined_model_dict.ipynb.
Another configuration is the combination of T5 with a trained idiom model. This configuration is in combined_model.ipynb
### Streamlit
To run the streamlit version of this repository, please install streamlit module first:
```bash
pip install streamlit 
```
To run the app, please use following commands
```bash
cd streamlit_app
streamlit run app.py
```
Note that you must run both commands to prevent import (data not found) error.

### Data 
All the data is stored in data subdir

Whenever the data for inference is changed, the vocab size of the idiom model is trained, this model must be retrained by running script in idiom_model_train.ipynb.
## License

[MIT](https://choosealicense.com/licenses/mit/)
