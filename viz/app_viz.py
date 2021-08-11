#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:43:16 2020

@author: aaron
"""

import streamlit as st
import pandas as pd
import numpy as np

import os
import json
import boto3
from botocore.client import Config
from annotated_text import annotated_text

st.beta_set_page_config(
    layout="wide",
)

def convert_streamlit_viz(doc):
    sentences = doc['sentences']
    events_doc = doc['predicted_events']
    result = []
    offset=0
    new_sent_list=[]
    for sent, events_sent in zip(sentences, events_doc):
        sent = [' '+token+' ' for token in sent]
        for event in events_sent:
            for element in event:
                if len(element)>4: #argument
                    start = int(element[0])-offset
                    try:
                        end = int(element[1])-offset+1 # add 1 to the end
                        num_blank_tokens = end-start-1
                        # When we join the words together we need to preserve the token indices for the spans
                        blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]
                        argument = (' '+' '.join([token[0] if isinstance(token, tuple()) else token for token in sent[start:end]])+' ', ' '+element[2])                        
                        sent = sent[:start]+[argument]+blank_tokens+sent[end:]

                    except:
                        end = int(element[1])-offset+1 # add 1 to the end
                        num_blank_tokens = end-start-1
                        # When we join the words together we need to preserve the token indices for the spans
                        blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]
                        argument = (' '+' '.join([token[0] if isinstance(token, tuple()) else token[0] for token in sent[start:end]])+' ', ' '+element[2])
                        sent = sent[:start]+[argument]+blank_tokens+sent[end:]
                    
                else: #trigger
                    start = int(element[0])-offset
                    end = int(element[0])-offset+1 # add 1 to the end
                    num_blank_tokens = end-start-1
                    # When we join the words together we need to preserve the token indices for the spans
                    blank_tokens = [['<blank>'] for _ in range(num_blank_tokens)]    
                    trigger = (' '+' '.join(sent[start:end][0])+' ', ' '+element[1], "#afa")
                    sent = sent[:start]+[trigger]+blank_tokens+sent[end:]

        new_sent_list.append([token for token in sent if token != ['<blank>']])
        offset=offset+len(sent)
    return tuple(new_sent_list)


def display_annotations(preds):
    for idx, examples in enumerate(preds):
        example = json.loads(examples)
        input_list = []
        st.header('Batch {}:'.format(idx))
        for sample in convert_streamlit_viz(example):            
            
            if isinstance(sample, list):
                input_list=input_list+(sample)
            else:   
                input_list=input_list+sample
            
            if len(input_list)>80:
                annotated_text(input_list)
                input_list=[]
    
        annotated_text(input_list)

   

st.header('Prediction Visualization')
predictions_file = st.file_uploader("Upload predictions", type=["jsonl"])  
if predictions_file:
    preds=predictions_file.readlines()
    display_annotations(preds)

