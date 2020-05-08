# bert_physics

## Installation
1. Install the dependencies in your venv with:
    ```
    pip3 install -r requirements.txt
    ```
2. If the transformers package is not installed through pip, clone it from git:
    ```
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .
    ```
## Index
1. Files:
    * *bert_exp.py*: transformers + BERT experiments to calculate entropy and perplexity
    * *viz.py*: script to create result visualizations
    * *process_survey.py*: to preprocess the human survey results
2. Data:
    * *bert_options.csv*: the input of **bert_exp.py**
    * *bert_result.csv*: the output of **bert_exp.py**
3. Figs: a folder to keep the figures made by **viz.py**
