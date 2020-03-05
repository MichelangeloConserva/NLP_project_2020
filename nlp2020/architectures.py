import torch.nn as nn
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig


class NLP_NN(nn.Module):
    
    def __init__(self, outputs):
        super(NLP_NN, self).__init__()
        
        self.outputs = outputs

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        
        # TODO : activate back for training
        # if torch.cuda.is_available(): self.model.cuda()

        self.sm = torch.nn.Softmax(dim=1)


    def forward(self, x):
        x = x.long()
        x = self.model(x)[0]
        
        return self.sm(x)
    
    
    
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()

        self.hl1 = nn.Linear(inputs, 256)
        self.hl2 = nn.Linear(256, 128)
        self.hl3 = nn.Linear(128, 64)
        self.hl4 = nn.Linear(64, outputs)
        self.distribution = torch.distributions.Categorical
        
        # if torch.cuda.is_available(): self.cuda()
        
    def forward(self, x):
        x = self.hl1(x)
        x = torch.tanh(self.hl2(x))
        x = torch.tanh(self.hl3(x))
        x = self.hl4(x)
        
        return x
