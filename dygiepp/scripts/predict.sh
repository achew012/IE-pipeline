#!/bin/bash
model_name=$1
test_filepath=$2 

mkdir "results/${model_name}"

echo "Evaluating ${model_name} on test dataset..."

allennlp evaluate "models/${model_name}" \
  "data/${test_filepath}" \
  --cuda-device 0 \
  --include-package dygie \
  --output-file "results/${model_name}/output_metrics.json" & # Optional; if not given, prints metrics to console.

allennlp predict "models/${model_name}" \
    "data/${test_filepath}" \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file "results/${model_name}/predictions.jsonl" \
    --cuda-device 0 \
    --silent \

echo "Scores and predictions can be found at results/${model_name}"

