from json import load
import os 
import json 
import re 
from collections import defaultdict 
import argparse 
import shutil

import transformers 
from transformers import BartTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 

from .data import IEDataset, my_collate
from .utils import load_ontology

MAX_LENGTH=170
MAX_TGT_LENGTH=72

class ACEDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>'])
    

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True, index=0):
        '''
        If there are multiple events per example, use index parameter.

        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found. 
        '''

        evt_type = ex['event_mentions'][index]['event_type']

        context_words = ex['tokens']
        template = ontology_dict[evt_type]['template']
        input_template = re.sub(r'<arg\d>', '<arg>', template) 


        space_tokenized_input_template = input_template.split()
        tokenized_input_template = [] 
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        
        role2arg = defaultdict(list)

        for argument in ex['event_mentions'][index]['arguments']:
            role2arg[argument['role']].append(argument)

        role2arg = dict(role2arg)
        arg_idx2text = defaultdict(list)
        for role in role2arg.keys():
            if role not in ontology_dict[evt_type]:
                # annotation error 
                continue 
            for i, argument in enumerate(role2arg[role]):
                arg_text = argument['text']
                if i < len(ontology_dict[evt_type][role]):
                    # enough slots to fill in 
                    arg_idx = ontology_dict[evt_type][role][i]
                    
                
                else:
                    # multiple participants for the same role 
                    arg_idx = ontology_dict[evt_type][role][-1]

                arg_idx2text[arg_idx].append(arg_text)
                
        for arg_idx, text_list in arg_idx2text.items():
            text = ' and '.join(text_list)
            template = re.sub('<{}>'.format(arg_idx), text, template)            

        trigger = ex['event_mentions'][index]['trigger']
        # trigger span does not include last index 

        if mark_trigger:
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger['start']]), add_prefix_space=True) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger['start']: trigger['end']]), add_prefix_space=True)
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger['end']:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        output_template = re.sub(r'<arg\d>','<arg>', template ) 
        space_tokenized_template = output_template.split()
        tokenized_template = [] 
        for w in space_tokenized_template:
            tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        
        return tokenized_input_template, tokenized_template, context

                
    def prepare_data(self):
        if self.hparams['tmp_dir']:
            data_dir = self.hparams['tmp_dir']
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams['dataset'])
        
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

        if not os.path.exists(data_dir):
            print('creating tmp dir ....')
            os.makedirs(data_dir)

            ontology_dict = load_ontology(dataset='KAIROS')
            ontology_dict = {key.replace('.', ':'):ontology_dict[key] for key in ontology_dict.keys()}

            with open(self.hparams['test_file'],'r') as reader,  open(os.path.join(data_dir,'{}.jsonl'.format('tmp')), 'w') as writer:
                for lidx, line in enumerate(reader):
                    ex = json.loads(line.strip())

                    for i in range(len(ex['event_mentions'])):
                        evt_type = ex['event_mentions'][i]['event_type']

                        if evt_type not in ontology_dict: # should be a rare event type 
                            print('**', evt_type)
                            continue 
                        
                        input_template, output_template, context= self.create_gold_gen(ex, ontology_dict, self.hparams['mark_trigger'], index=i)
                                                
                        input_tokens = self.tokenizer.encode_plus(input_template, context, 
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=MAX_LENGTH,
                                truncation='only_second',
                                padding='max_length')
                        
                        tgt_tokens = self.tokenizer.encode_plus(output_template, 
                        add_special_tokens=True,
                        add_prefix_space=True, 
                        max_length=MAX_TGT_LENGTH,
                        truncation=True,
                        padding='max_length')

                        processed_ex = {
                            'doc_key': ex['sent_id'], #this is not unique 
                            'input_token_ids':input_tokens['input_ids'],
                            'input_attn_mask': input_tokens['attention_mask'],
                            'tgt_token_ids': tgt_tokens['input_ids'],
                            'tgt_attn_mask': tgt_tokens['attention_mask'],
                        }
                        
                        writer.write(json.dumps(processed_ex) + '\n')

    
    def train_dataloader(self):
        if self.hparams['tmp_dir']:
            data_dir = self.hparams['tmp_dir']
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams['dataset'])

        dataset = IEDataset(os.path.join(data_dir, 'tmp.jsonl'))
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=4, 
            collate_fn=my_collate,
            batch_size=self.hparams['train_batch_size'], 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        if self.hparams['tmp_dir']:
            data_dir = self.hparams['tmp_dir']
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams['dataset'])

        dataset = IEDataset(os.path.join(data_dir, 'tmp.jsonl'))
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=4, 
            collate_fn=my_collate,
            batch_size=self.hparams['eval_batch_size'], shuffle=False)
        return dataloader

    def test_dataloader(self):
        if self.hparams['tmp_dir']:
            data_dir = self.hparams['tmp_dir']
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams['dataset'])

        dataset = IEDataset(os.path.join(data_dir, 'tmp.jsonl'))

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=4, 
            collate_fn=my_collate, 
            batch_size=self.hparams['eval_batch_size'], shuffle=False)

        return dataloader


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser() 
#     parser.add_argument('--train-file',type=str)
#     parser.add_argument('--val-file', type=str)
#     parser.add_argument('--test-file', type=str)
#     parser.add_argument('--tmp_dir', default='tmp')
#     parser.add_argument('--train_batch_size', type=int, default=2)
#     parser.add_argument('--eval_batch_size', type=int, default=4)
#     parser.add_argument('--mark-trigger', action='store_true', default=True)
#     parser.add_argument('--dataset', type=str, default='combined')
#     args = parser.parse_args() 

#     dm = ACEDataModule(args=args)
#     dm.prepare_data() 

#     # training dataloader 
#     dataloader = dm.test_dataloader() 

# #     for idx, batch in enumerate(dataloader):
# #         print(batch)
# #         break 

    