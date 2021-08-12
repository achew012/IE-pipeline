from fastapi import FastAPI, Request
from utils import *
import requests
import json
from fastapi.exceptions import HTTPException
from asyncio import Lock, sleep
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE


#service_state={'predicting':False}
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', do_lower_case=False)                                               
lock = Lock()
app = FastAPI(title="FastAPI")

def predict_triggers(dataset:list):
    response = requests.post('http://dygiepp:5001/predict', json = dataset)
    return response.json()


def predict_arguments(dataset):
    ontology = json.load(open('event_role_KAIROS.json'))
    response = requests.post('http://genarg:5002/predict', json = dataset)
    response = response.json()
    response = [json.loads(res) for res in response[0]['result']]
    return response, ontology

def predict_templates(dataset):
    response = requests.post('http://gtt:5003/predict', json = dataset)
    response = response.json()
    return response

@app.get("/")
def main():
    print('Attempting to predict...')
    try:
        # Pass wikievents data to dygie format to extract triggers
        wikievent = load_jsonl('/data/wikievents/{}.jsonl'.format('test'))
        wikievent_in_dygiepp_fmt = wikievents2dygie(wikievent, 'test')
        wikievent_triggers = predict_triggers(wikievent_in_dygiepp_fmt)
        wikievent_triggers = [{key:doc[key] for key in doc.keys() if key not in ['ner', 'events']} for doc in wikievent_triggers]
        to_jsonl('/outputs/triggers.jsonl', wikievent_triggers)

        ### Dygiepp to Wikievents approach (KIV - take dygiepp triggers and entities and merge them with the wikievent dict)
        # TRIGGER {"id": "road_ied_8-E1", "event_type": "Life.Die.Unspecified", "trigger": {"start": 2, "end": 3, "text": "kills", "sent_idx": 0}, "arguments": [{"entity_id": "road_ied_8-T3", "role": "Victim", "text": "general"}, {"entity_id": "road_ied_8-T4", "role": "Place", "text": "Syria"}]}
        # ENTITY {"id": "road_ied_8-T1", "sent_idx": 0, "start": 1, "end": 2, "entity_type": "WEA", "mention_type": "UNK", "text": "IED"}
        #wikievent_triggers = [{'doc_key': doc['doc_key'], 'sentences': doc['sentences'], 'triggers': doc['predicted_events'], 'entities': doc['predicted_ner']} for doc in wikievent_triggers]

        wikievent_in_oneie_fmt = dygiepp2oneie(wikievent_triggers, bart_tokenizer)
        wikievent_args, ontology = predict_arguments(wikievent_in_oneie_fmt)

        # print('Grouping Arg Preds...')
        preds = group_preds(wikievent_args)

        # print('Consolidating Preds...')
        output, unique_role_templates = combine_data_and_preds(wikievent, ontology, preds)    
        to_jsonl('/outputs/arguments.jsonl', output)

        print('Prediction Completed')
        return output

    except Exception as e:
        print('ERROR: ', e)
        return {'predicted': False}
