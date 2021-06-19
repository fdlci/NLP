import numpy as np
import pandas as pd
from torch.utils.data import Dataset



class DataPrep(Dataset):
    def __init__(self, filedata,tokenizer,label_to_id = None, category_to_id=None):
        super(DataPrep, self).__init__()
        self.tokenizer = tokenizer
        self.filedata = filedata
        columns = ['opinion', 'categories', 'word', 'position', 'sentence']
        self.data = pd.read_csv(filedata, sep = '\t', names=columns)
        
        if label_to_id is not None :
            self.label_to_id = label_to_id
        else :
            possible_labels = self.data.opinion.unique().tolist()
            self.label_to_id = {label:k for k, label in enumerate(possible_labels)}

        if category_to_id is not None :
            self.category_to_id = category_to_id 
        else :
            possible_categories = self.data.categories.unique().tolist()
            self.category_to_id = {label:k for k, label in enumerate(possible_categories)}
        
        self.id_to_label = {id:label for label, id in self.label_to_id.items()}
        self.id_to_category = {id:category for category, id in self.category_to_id.items()}
        
        #self.data = self.dftoid(self.data)

        opinion = self.data.opinion.tolist()
        sentences = self.data.sentence.tolist()
        position = self.data.position.tolist()
        categories = self.data.categories.tolist()

        sentences_tokenized = []
        ind_words_to_guess = []
        labels = []
        category_list = []
        self.token_id_begin, self.token_id_end, self.token_id_pad = self.tokenizer("", return_token_type_ids=False, return_attention_mask=False, padding="max_length", max_length=3)['input_ids']


        for i, s in enumerate(sentences):
            interval = position[i].split(":")
            ind0, ind1 = int(interval[0]), int(interval[1])
            tokens, ind_start, ind_end = self.convert_sentence_to_tokens(s, ind0, ind1)
            sentences_tokenized.append(tokens)
            ind_words_to_guess.append([ind_start, ind_end])
            labels.append(self.label_to_id[opinion[i]])
            category_list.append(self.category_to_id[categories[i]])

        self.sentences_tokenized = sentences_tokenized
        self.ind_words_to_guess = ind_words_to_guess
        self.labels = labels
        self.category_list = category_list




    def __getitem__(self, index) :
        return self.sentences_tokenized[index], self.ind_words_to_guess[index], self.labels[index], self.category_list[index]
        
    def __len__(self):
        return len(self.labels)


    # def dftoid(self, df):
        
    #     df['opinion'] = [self.label_to_id[label] for label in df.opinion]
    #     df['categories'] = [self.category_to_id[cat] for cat in df.categories]
    #     return df

    def convert_sentence_to_tokens(self, sentence, ind0, ind1):
        tokens_list = [self.token_id_begin]
        if ind0 > 0 : 
            tokens_list += self.tokenizer.encode_plus(sentence[:ind0],add_special_tokens=True, return_token_type_ids =False, return_attention_mask =False)['input_ids'][1:-1]
            
        ind_start = len(tokens_list)
        tokens_list += self.tokenizer.encode_plus(sentence[ind0:ind1],add_special_tokens=True, return_token_type_ids =False, return_attention_mask =False)['input_ids'][1:-1]
        ind_end = len(tokens_list)

        if ind1 < len(sentence) :
            tokens_list += self.tokenizer.encode_plus(sentence[ind1:], add_special_tokens=True,return_token_type_ids =False, return_attention_mask =False)['input_ids'][1:-1]

        tokens_list += [self.token_id_end]

        return tokens_list, ind_start, ind_end

