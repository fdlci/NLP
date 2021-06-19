from Data_prepa import DataPrep
import numpy as np
import torch.nn as nn
from torch import optim
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from sentbert import BertClassifier
import copy


def custom_collate_fn(batch, pad_id):
    length_to_pad = max([len(item[0]) for item in batch])
    sentences = []
    ind_words_to_guess = []
    labels = []
    category_list = []
    for item in batch :
      sentences.append(item[0] + (length_to_pad-len(item[0])) * [pad_id])
      ind_words_to_guess.append(item[1])
      labels.append(item[2])
      category_list.append(item[3])
    return torch.tensor(sentences), torch.tensor(ind_words_to_guess), torch.tensor(labels), torch.tensor(category_list)

def train_epch(dataloader, criterion, optimizer, model, device,scheduler, epch):
    model.train()
    train_loss = 0.
    for _, (data, words_id, labels, categories) in enumerate(dataloader):
        data = data.to(device)
        words_id = words_id.to(device)      
        labels = labels.to(device)
        categories = categories.to(device)

        outputs = model({"data":data, "words_id":words_id, "categories":categories})

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    print("Training done epoch {}!     Loss : {}".format(epch, round(train_loss,3)))

def eval_epch(dataloader,criterion, model, device):
    model.eval()
    correct = 0
    total = 0
    predicted_list = []
    with torch.no_grad():
        for _, (data, words_id, labels, categories) in enumerate(dataloader):
            data = data.to(device)
            words_id = words_id.to(device)      
            categories = categories.to(device)
            labels = labels.to(device)

            outputs = model({"data":data, "words_id":words_id, "categories":categories})

            _, predicted = torch.max(outputs, 1)
            predicted_list.append(int(predicted))

            total += labels.size(0) 
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

    print("Evaluation done! Accuracy : {}%".format(round(correct/total*100,1)))
    return  loss, predicted_list


class Classifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()

        #attribute that will be filled during training
        self.label_to_id = None
        self.category_to_id = None


    def initialize_model(self,train_dataloader, epochs=4, catego = True, freeze_bert=False):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        self.model = BertClassifier().to(self.device)

        # Tell PyTorch to run the model on GPU
        self.model.to(self.device)

        # Create the optimizer
        self.optimizer = AdamW(self.model.parameters(),
                        lr=8e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=4, # Default value
                                                    num_training_steps=total_steps)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        #return bert_classifier, optimizer, scheduler



    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        batch_size = 32
        epochs = 3
        train_dataset = DataPrep(trainfile, self.tokenizer)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn= lambda b: custom_collate_fn(b, train_dataset.token_id_pad))
        #val_dataset = DataPrep("../data/devdata.csv", self.tokenizer, self.label_to_id, self.category_to_id)
        #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn= lambda b: custom_collate_fn(b, val_dataset.token_id_pad))

        self.label_to_id = train_dataset.label_to_id
        self.category_to_id = train_dataset.category_to_id
        self.initialize_model(train_dataloader,epochs)
        #optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #criterion = nn.CrossEntropyLoss()
        best_loss = np.inf

        for epch in range(epochs):
            train_epch(train_dataloader, self.loss_fn, self.optimizer, self.model, self.device,self.scheduler, epch)
        #     loss_eval,_ = eval_epch(val_dataloader,self.loss_fn, self.model, self.device) 
        #     print('Evaluation done: loss: '+str(loss_eval.item()))
        #     if loss_eval<best_loss:
        #       best_loss = loss_eval
        #       best_model = copy.deepcopy(self.model.state_dict())
        
        # self.model.load_state_dict(best_model)

            
    
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        val_dataset = DataPrep(datafile, self.tokenizer, self.label_to_id, self.category_to_id)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn= lambda b: custom_collate_fn(b, val_dataset.token_id_pad))
        _, predicted_list = eval_epch(val_dataloader,self.loss_fn, self.model, self.device)
        print(predicted_list)
        return [val_dataset.id_to_label[pred] for pred in predicted_list]

        
        

