local template = import "template.libsonnet";

#allenai/longformer-base-4096

template.DyGIE {
  bert_model: "bert-base-uncased",
  max_wordpieces_per_sentence: 1024,
  cuda_device: 0,
  data_paths: {
    train: "data/wikievents/processed-data/train.jsonl",
    validation: "data/wikievents/processed-data/dev.jsonl",
    test: "data/wikievents/processed-data/test.jsonl",

    #collation screws up the format
    #train: "data/wikievents/normalized-data/train.jsonl",
    #validation: "data/wikievents/normalized-data/dev.jsonl",
    #test: "data/wikievents/normalized-data/test.jsonl",

  },
  loss_weights: {
    ner: 1,
    relation: 0.0,
    coref: 0.0,
    events: 1.0
  },

  target_task: "events"
}
