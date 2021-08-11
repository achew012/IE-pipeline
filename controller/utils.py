import json
import jsonlines
import re
from collections import defaultdict 
from tqdm import tqdm
from transformers import (
    BartModel,
    BartTokenizer
)

def load_jsonl(load_path:str):
    data = []
    with open(load_path, 'r', encoding='ascii') as file:
        for doc in file:
            data.append(json.loads(doc))
    return data

def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj) 

def group_preds(preds: list):
    grouped_preds = {}    
    for pred in preds:
        doc_key = pred['doc_key'].split('-')[0]
        pred = {**pred, 'event': pred['doc_key'].split('-')[1]}
        if pred['doc_key'] not in grouped_preds.keys():
            grouped_preds[doc_key]=[pred]
        else:
            grouped_preds[doc_key].append(pred)
    return grouped_preds

def extract_args_from_template(ex, template, evt_type, ontology_dict):
    # extract argument text 
    template_words = template.strip().split()
    predicted_words = ex['predicted'].strip().split()    
    predicted_args = defaultdict(list) # each argname may have multiple participants 
    t_ptr= 0
    p_ptr= 0 
        
    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)

            # Get role name
            try:
                if evt_type in ontology_dict.keys():
                    #print(ontology_dict[evt_type]['roles'])
                    #print(arg_num)
                    arg_name = ontology_dict[evt_type]['roles'][int(arg_num[-1])-1]
                else:
                    arg_name=''
            except KeyError:
                print(KeyError)
                print(evt_type)
                #exit() 

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and ((t_ptr== len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                    p_ptr+=1 
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name].append(arg_text)
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    
    return predicted_args[0]

def extract_args_from_pred_templates(data, ontology):
    for doc_idx, doc in enumerate(data):
        if doc['event_preds']:
            data[doc_idx]['extracted_arg_preds'] = []
            for example, template, evt_mention in zip(doc['event_preds'], doc['event_templates'], doc['event_mentions']):
                args = extract_args_from_template(example, template, evt_mention['event_type'], ontology)   
                data[doc_idx]['extracted_arg_preds'].append(args)
    return data                

def make_roles_for_template_filling(arguments, roles):
    template = {role:[] for role in roles}
    for arg in arguments:
        if arg['role'] in template.keys():
            template[arg['role']].append(arg['text'])
    return template

def combine_data_and_preds(data, ontology, preds=None):
    events = []
    templates = []
    for doc_idx, doc in enumerate(data):
        if preds!=None:
            if data[doc_idx]['doc_id'] in preds.keys():
                data[doc_idx]['event_preds'] = preds[doc['doc_id']]
            else:
                data[doc_idx]['event_preds'] = []

        data[doc_idx]['event_templates'] = []
        data[doc_idx]['role_templates'] = []
        for evt_idx, event in enumerate(doc['event_mentions']):
            
            if event['event_type'] in ontology.keys():
                role_template = make_roles_for_template_filling(event['arguments'], ontology[event['event_type']]['roles'])                
                sequence_template = ontology[event['event_type']]['template']                
                data[doc_idx]['role_templates'].append(role_template)
                data[doc_idx]['event_templates'].append(sequence_template)
                #templates.append(sequence_template)
                templates.append(','.join(list(role_template.keys())))
            else:
                data[doc_idx]['role_templates'].append({})
                data[doc_idx]['event_templates'].append('')      

    unique_role_templates = list(set(templates))

    if preds!=None:
       data = extract_args_from_pred_templates(data, ontology)
       #data = [{key:doc[key] for key in doc.keys() if key not in ['event_mentions', 'role_templates']} for doc in data]

    return data, unique_role_templates


def wikievents2dygie(wikievent:list, split:str, maxlength=1024):
    entity_list = [{entity['text']:[entity['start'], entity['end']] for entity in doc['entity_mentions']} for doc in wikievent]
    wikievent = [{'dataset': "wikievent", 'doc_key': doc['doc_id'], 'doc_len': len(doc['tokens']+['.']), 'sentences': [doc['tokens']+['.']], '_sentence_start': [0], 'events': [[
        [[event['trigger']['start'], event['event_type']]]
        +[entities[arg['text']]+[arg['role']] if arg['text'] in entities.keys() else ['0','0', arg['text'], arg['role']] for arg in event['arguments']] for event in doc['event_mentions']
        ]], 
        'ner':[[[entity['start'], entity['end'], entity['entity_type']] for entity in doc['entity_mentions']]]} for entities, doc in zip(entity_list, wikievent)]
    
    wikievent = [{**event, 'sentences': [[token.encode('utf-8').decode('ascii', errors='ignore') for token in event['sentences'][0]]]} for event in wikievent]        

    wikievent = [event for event in wikievent if event.pop('doc_len')<=maxlength]    

    return wikievent

def dygie2wikievents(wikievent:list, split:str):
    return wikievent

def map_index(pieces):
    idxs = []
    for i, piece in enumerate(pieces):
        if i == 0:
            idxs.append([0, len(piece)])
        else:
            _, last = idxs[-1]
            idxs.append([last, last + len(piece)])
    return idxs


def dygiepp2oneie(dygiepp, tokenizer):
    data = []    
    for doc in tqdm(dygiepp):
        doc_id = doc['doc_key']
        sentences = doc['sentences']
        sent_num = len(sentences)
        entities = doc.get('predicted_ner', [[] for _ in range(sent_num)])
        relations = doc.get('relations', [[] for _ in range(sent_num)])
        events = doc.get('predicted_events', [[] for _ in range(sent_num)])
    
        offset = 0
        for i, (sent_tokens, sent_entities, sent_relations, sent_events) in enumerate(zip(
            sentences, entities, relations, events
        )):
            sent_id = '{}-{}'.format(doc_id, i)
            pieces = [tokenizer.tokenize(t) for t in sent_tokens]
            word_lens = [len(p) for p in pieces]
            idx_mapping = map_index(pieces)
    
            sent_entities_ = []
            sent_entity_map = {}
            for j, (start, end, entity_type, _, _) in enumerate(sent_entities):
                start, end = start - offset, end - offset + 1
                entity_id = '{}-E{}'.format(sent_id, j)
                entity = {
                    'id': entity_id,
                    'start': start, 'end': end,
                    'entity_type': entity_type,
                    # Mention types are not included in DyGIE++'s format
                    'mention_type': 'UNK',
                    'text': ' '.join(sent_tokens[start:end])}
                sent_entities_.append(entity)
                sent_entity_map[start] = entity
    
            sent_relations_ = []
            for j, (start1, end1, start2, end2, rel_type) in enumerate(sent_relations):
                start1, end1 = start1 - offset, end1 - offset
                start2, end2 = start2 - offset, end2 - offset
                start2, end2 = start2 - offset, end2 - offset
                arg1 = sent_entity_map[start1]
                arg2 = sent_entity_map[start2]
                relation_id = '{}-R{}'.format(sent_id, j)
                rel_type = rel_type.split('.')[0]
                relation = {
                    'relation_type': rel_type,
                    'id': relation_id,
                    'arguments': [
                        {
                            'entity_id': arg1['id'],
                            'text': arg1['text'],
                            'role': 'Arg-1'
                        },
                        {
                            'entity_id': arg2['id'],
                            'text': arg2['text'],
                            'role': 'Arg-2'
                        },
                    ]
                }
                sent_relations_.append(relation)
    
            sent_events_ = []
            for j, event in enumerate(sent_events):
                event_id = '{}-EV{}'.format(sent_id, j)
                trigger_start, event_type, _, _ = event[0]
                trigger_end = trigger_start
                trigger_start, trigger_end = trigger_start - offset, trigger_end - offset + 1
                event_type = event_type.replace('.', ':')
                args = event[1:]
                args_ = []
                for arg_start, arg_end, role, _, _ in args:
                    if arg_start in sent_entity_map.keys():
                        arg_start, arg_end = arg_start - offset, arg_end - offset
                        arg = sent_entity_map[arg_start]
                        args_.append({
                            'entity_id': arg['id'],
                            'text': arg['text'],
                            'role': role
                        })
                event_obj = {
                    'event_type': event_type,
                    'id': event_id,
                    'trigger': {
                        'start': trigger_start,
                        'end': trigger_end,
                        'text': ' '.join(sent_tokens[trigger_start:trigger_end])
                    },
                    'arguments': args_
                }
                sent_events_.append(event_obj)
    
            sent_ = {
                'doc_id': doc_id,
                'sent_id': sent_id,
                'entity_mentions': sent_entities_,
                'relation_mentions': sent_relations_,
                'event_mentions': sent_events_,
                'tokens': sent_tokens,
                'pieces': [p for w in pieces for p in w],
                'token_lens': word_lens,
                'sentence': ' '.join(sent_tokens)
            }
            
            data.append(sent_)
            offset += len(sent_tokens)
    return data


