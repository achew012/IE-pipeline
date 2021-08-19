import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 
import shutil

import torch 
import wandb 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

import json
import jsonlines
from src.genie.data_module import RAMSDataModule
from src.genie.ACE_data_module import ACEDataModule
from src.genie.KAIROS_data_module import KAIROSDataModule 
from src.genie.model import GenIEModel 

#logger = logging.getLogger(__name__)

args = {
        'model': 'gen', 
        'dataset': 'ACE', 
        'tmp_dir': None, 
        'ckpt_name': 'gen-ACE-pred', 
        'load_ckpt': '/models/genarg.ckpt', 
        'train_file': 'intermediate.jsonl', 
        'val_file': 'intermediate.jsonl', 
        'test_file': 'intermediate.jsonl', 
        'input_dir': None, 
        'coref_dir': 'data/kairos/coref_outputs', 
        'use_info': False, 'mark_trigger': False, 
        'sample_gen': False, 
        'train_batch_size': 2, 
        'eval_batch_size': 2, 
        'eval_only': False, 
        'learning_rate': 5e-05, 
        'accumulate_grad_batches': 1, 
        'weight_decay': 0.0, 
        'adam_epsilon': 1e-08, 
        'gradient_clip_val': 1.0, 
        'num_train_epochs': 3, 
        'max_steps': -1, 
        'warmup_steps': 0, 
        'gpus': -1, 
        'seed': 42, 
        'fp16': False, 
        'threads': 1, 
        'ckpt_dir': './checkpoints/gen-ACE-pred'
        }

# Cold Start
model = GenIEModel(args)
if args['load_ckpt']:
    model.load_state_dict(torch.load(args['load_ckpt'] ,map_location=model.device)['state_dict']) 

def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)

def main(args, model):

    if not args['ckpt_name']:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args['ckpt_name'] = '{}_{}lr{}_{}'.format(args['model'],  args['train_batch_size'] * args['accumulate_grad_batches'], 
                args['learning_rate'], time_str)

    args['ckpt_dir'] = os.path.join(f'./checkpoints/{args["ckpt_name"]}')
    
    if os.path.exists(args['ckpt_dir']):
        shutil.rmtree(args['ckpt_dir'])
        os.makedirs(args['ckpt_dir'])
    else:    
        os.makedirs(args['ckpt_dir'])

    # args['ckpt_dir'] = os.path.join('./checkpoints/{}'.format(args['ckpt_name']))
    # if not os.path.exists(args['ckpt_dir']):
    #     os.makedirs(args['ckpt_dir'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['ckpt_dir'],
        save_top_k=2,
        monitor='val/loss',
        mode='min',
        save_weights_only=True,
        filename='{epoch}', # this cannot contain slashes 
    ) 

    # lr_logger = LearningRateMonitor() 
    # tb_logger = TensorBoardLogger('logs/')

    if args['dataset'] == 'ACE':
        dm = ACEDataModule(args)
    elif args['dataset'] == 'KAIROS':
        dm = KAIROSDataModule(args)

    if args['max_steps'] < 0 :
        args['max_epochs'] = args['min_epochs'] = args['num_train_epochs']    

    trainer = Trainer(
        logger=None,
        min_epochs=args['num_train_epochs'],
        max_epochs=args['num_train_epochs'], 
        gpus=args['gpus'], 
        checkpoint_callback=checkpoint_callback, 
        accumulate_grad_batches=args['accumulate_grad_batches'],
        gradient_clip_val=args['gradient_clip_val'], 
        num_sanity_val_steps=0, 
        val_check_interval=0.5, # use float to check every n epochs 
        precision=16 if args['fp16'] else 32,
        callbacks = None,
    ) 

    dm.prepare_data() 
    test_loader = dm.test_dataloader()

    #dm.setup('test')
    outputs = trainer.test(model, test_dataloaders=test_loader) #also loads training dataloader     
    return outputs    

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
        to_jsonl('intermediate.jsonl', data)
        if lock.locked():
            raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Service busy")
        async with lock:
            result = main(args, model)
            torch.cuda.empty_cache()
            return result
    else:
        torch.cuda.empty_cache()
        return None