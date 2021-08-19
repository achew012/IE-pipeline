import argparse
import os
import json
import spacy

import torch
from dygie.predictors import DyGIEPredictor
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Batch

import json
import jsonlines
from spacy import displacy


def display_results(results):
    formatted_results = convert_to_displacy(results)
    displacy.serve(formatted_results[0], manual=True, style='ent')


def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

''' Writes to a jsonl file'''
def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)

# from a file
def format_document(fname, nlp):
    text = open(fname).read()
    doc = nlp(text)
    sentences = [[tok.text for tok in sent] for sent in doc.sents]
    doc_key = fname
    #doc_key = os.path.basename(fname).replace(".txt", "")
    res = {
           "dataset": "ace-event",
           "doc_key": doc_key,
           "sentences": sentences,
           "ner":[[] for i in range(len(sentences))]
          }
    return res

# from variable
def format_text(fname, sentences, nlp, model_type):
    if isinstance(sentences, str):
        text = sentences
        doc = nlp(text)
        sentences = [[tok.text for tok in sent] for sent in doc.sents]
        doc_key = fname
        #doc_key = os.path.basename(fname).replace(".txt", "")
        res = {
               #"dataset": "ace-event",
               "dataset": model_type,
               "doc_key": doc_key,
               "sentences": sentences,
               "ner":[[] for i in range(len(sentences))]
              }
        return res
    else:
        print('error: ')
        print(sentences)
        return 'error'
# from variable
def format_object(title, data_object, nlp, model_type):
    res = format_text(title, data_object, nlp, model_type)
    return res

# from file
def format_datasets(data_directory, output_file, use_scispacy):
    nlp_name = "en_core_web_md"
    nlp = spacy.load(nlp_name)
    fnames = [f"{data_directory}/{name}" for name in os.listdir(data_directory)]
    res = [format_document(fname, nlp) for fname in fnames]
    return res

def get_model(model_filepath):
    import_module_and_submodules('dygie')
    archive = load_archive(model_filepath, 0)
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    model = DyGIEPredictor(archive.model, dataset_reader)
    return dataset_reader, model

def dygiepp_pretrained_predict(formatted_data, dataset_reader, model):
    # instances = dataset_reader.read(test_file)
    # batch = Batch(instances)
    # batch.index_instances(model.vocab)
    #for doc, gold_data in zip(iterator(batch.instances, num_epochs=1, shuffle=False), gold_test_data):
    instance_input = dataset_reader.text_to_instance(formatted_data)
    result = model.predict_instance(instance_input)
    return result

def spans(text, fragments):
    result = []
    point = 0  # Where we're in the text.
    for fragment in fragments:
        found_start = text.index(fragment, point)
        found_end = found_start + len(fragment)
        result.append((found_start, found_end))
        point = found_end
    return result

def convert_to_displacy(doc):
    sentences = doc['sentences']
    events_doc = doc['predicted_events']
    result = []
    offset=0
    for sent, events_sent in zip(sentences, events_doc):
        span_list = spans(' '.join(sent), sent)
        trig_or_arg = []
        for event in events_sent:
            for element in event:
                if len(element)>4: #argument
                    if int(element[0])>len(span_list):
                        idx_start = len(span_list)-1
                    if int(element[1])>len(span_list):
                        idx_end = len(span_list)-1
                    if int(element[0])<len(span_list) and int(element[1])<len(span_list):
                        idx_start = int(element[0])-offset
                        idx_end = int(element[1])-offset
                    trig_or_arg.append({"start": span_list[idx_start][0], "end": span_list[idx_end][1], "label": element[2]})

                else: #trigger
                    if int(element[0])>len(span_list):
                        idx_start = len(span_list)-1
                        idx_end = len(span_list)-1
                    if int(element[0])<len(span_list):
                        idx_start = int(element[0])-offset
                        idx_end = int(element[0])-offset
                    trig_or_arg.append({"start": span_list[idx_start][0], "end": span_list[idx_end][1], "label": element[1]})
        result.append({
        "text": ' '.join(sent),
        "ents": trig_or_arg,
        "title": None
        })
        offset=offset+len(sent)
    return result

def run_single_text_flow(input_text:str, pretrained_model_path:str, title='no-title', spacy_model="en_core_web_md", model_type="ace-event"):
    nlp = spacy.load(spacy_model)
    formatted_data = format_text(title, input_text, model_type, nlp, model_type)
    dataset_reader, model = get_model(pretrained_model_path)
    results = dygiepp_pretrained_predict(formatted_data, dataset_reader, model)
    return results


def run_dataset(dataset, model, model_type, dataset_reader, nlp):
    print("Detected {} documents.".format(len(dataset)))
    output = []
    for idx, doc in enumerate(dataset):
        print("Document {}".format(idx))  
        #formatted_data = format_object(doc['doc_key'], ' '.join(doc['sentences'][0]), nlp, model_type)
        results = dygiepp_pretrained_predict(doc, dataset_reader, model)
        output.append(results)
    return output

# Run cold start
spacy_model_type="en_core_web_md"
spacy_model = spacy.load(spacy_model_type)
model_type="wikievent"
pretrained_model_path='/models/dygiepp.tar.gz'
dataset_reader, model = get_model(pretrained_model_path)

####################################### EXAMPLES ##############################################
# text='''
# Jacinda Ardern has won a second term as New Zealand's Prime Minister after her success at handling the country's coronavirus outbreak helped secure a landslide victory.Preliminary results show that Ardern's center-left Labour Party has won 49% of the vote, meaning her party looks likely to score the highest result that any party has achieved since the current political system was introduced in 1996. That result means her party is projected to win 64 out of 120 parliamentary seats, making it the first party to be able to govern alone under the current system. Coalitions are the norm in New Zealand, where no single party has won a majority of votes in the last 24 years.
# ''' 
#ct_raw = read_json('ct.json')
#results = run_dataset(ct_raw, pretrained_model_path='../data/dygiepp/models/acepp_robertabase.tar.gz', spacy_model="en_core_web_md", model_type="ace-event-plus-rams")
#results = run_single_text_flow(text, pretrained_model_path='../data/dygiepp/models/ace-event.tar.gz', spacy_model="en_core_web_md", model_type="ace-event")
#results = run_single_text_flow(text, pretrained_model_path='../data/dygiepp/models/acepp_robertabase.tar.gz', spacy_model="en_core_web_md", model_type="ace-event-plus-rams")
#print(results)
#display_results(results)
#to_jsonl("ct_preds.jsonl", results)
################################################################################################

from fastapi import FastAPI, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi.exceptions import HTTPException
from asyncio import Lock, sleep
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE


class RequestBody(BaseModel):
    doc: list

lock = Lock()
app = FastAPI()

@app.post("/predict")
#async def read_root(body: RequestBody):
async def read_root(request: Request):
    data = await request.json()

    if data:
        if lock.locked():
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Service busy")
        async with lock:
            outputs = run_dataset(data, model, model_type, dataset_reader, spacy_model)
            outputs = [doc for doc in outputs if "_FAILED_PREDICTION" not in doc.keys()]
            torch.cuda.empty_cache()
        return outputs
    else:
        torch.cuda.empty_cache()
        return {}

