import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        
      

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.common_linear = nn.Linear(2*768,768)

        for i in range(12):
          setattr(self, 'linear_%i' % i, nn.Linear(768, 3))

    def forward(self, data):
        features = self.bert(data["data"])[0]
        last_hidden_state_cls = features[:, 0, :]
        output = []
        for k, feat in enumerate(features) :
            mean_feat = torch.mean(feat[data["words_id"][k][0]:data["words_id"][k][1],:], dim=0)
            mean_feat = torch.cat((mean_feat, last_hidden_state_cls[k,:]),0)
            mean_feat = torch.relu(self.common_linear(mean_feat))
            out = getattr(self, 'linear_' + str(int(data["categories"][k])))(mean_feat)
            output.append(out)

        return torch.stack(output)