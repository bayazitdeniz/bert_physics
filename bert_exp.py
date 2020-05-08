import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
loss_fn = nn.CrossEntropyLoss()


# a1 = "Jim kicked the ball, and Jim is stronger than the ball"
# b1 = "Jim kicked the ball, and the ball is stronger than Jim"
# a2 = 'The ship carried the car, and the ship is heavier than the car'
# b2 = 'The cargo carried the car, and the car is heavier than the cargo'

def evaluate_sentence(s):
    """
    Takes in the sentence *s* and calculates its cross entropy
    """
    x = tokenizer.encode_plus(s)
    input_token_ids = torch.tensor(x['input_ids'])
    with torch.no_grad():
        # Convert inputs to PyTorch tensors
        outputs = model(input_token_ids.unsqueeze(0), token_type_ids=torch.tensor([x['token_type_ids']]))
        predictions = outputs[0]
    crossentropy = loss_fn(predictions.squeeze(), input_token_ids).item()
    return crossentropy

def winner(perp_a, perp_b):
    """
    Decides who wins. A? B? Who knows.
    """
    if perp_a < perp_b:
        return "a"
    if perp_a > perp_b:
        return "b"
    if perp_a == perp_b:
        return "draw"

def main():
    df = pd.read_csv("data/bert_options.csv")
    df["a_crossentropy"] = df["option_a"].apply(lambda s: evaluate_sentence(s)) # default axis=0
    df["b_crossentropy"] = df["option_b"].apply(lambda s: evaluate_sentence(s))
    df["a_perplexity"] = df["a_crossentropy"].apply(lambda c: 2.0 ** c)
    df["b_perplexity"] = df["b_crossentropy"].apply(lambda c: 2.0 ** c)

    df["winner"] = df.apply(lambda row: winner(row.a_perplexity, row.b_perplexity), axis=1)
    df.to_csv("data/bert_result.csv", index=False)

if __name__ == '__main__':
    main()
