local template = import "template.libsonnet";

#allenai/longformer-base-4096

template.DyGIE {
  bert_model: "albert-large-v1",
  max_wordpieces_per_sentence: 1024,
  cuda_device: 0,
  data_paths: {
    #train: "data/ace-event/collated-data/test/json/train.json",
    #validation: "data/ace-event/collated-data/test/json/dev.json",
    #test: "data/ace-event/collated-data/test/json/test.json",
    train: "data/wikievents/train.jsonl",
    validation: "data/wikievents/dev.jsonl",
    test: "data/wikievents/test.jsonl",
  },
  loss_weights: {
    ner: 0.5,
    relation: 0.5,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events"
}
