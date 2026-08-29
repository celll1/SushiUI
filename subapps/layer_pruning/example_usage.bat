@echo off
REM Example usage for layer pruning tool (Windows)

echo Step 1: Extracting samples from dataset...
"%~dp0..\..\venv\Scripts\python.exe" subapps/layer_pruning/extract_samples.py ^
    --dataset-db datasets.db ^
    --dataset-id "your_dataset_unique_id" ^
    --num-samples 10 ^
    --output subapps/layer_pruning/samples.json

echo Step 2: Running greedy layer pruning (30 -^> 20 layers)...
"%~dp0..\..\venv\Scripts\python.exe" subapps/layer_pruning/prune_layers.py ^
    --model-path "path/to/your/model.safetensors" ^
    --samples subapps/layer_pruning/samples.json ^
    --target-layers 20 ^
    --strategy greedy ^
    --output subapps/layer_pruning/pruned_model_20layers.safetensors

echo Step 3: Evaluating pruned model...
"%~dp0..\..\venv\Scripts\python.exe" subapps/layer_pruning/evaluate_pruned.py ^
    --original-model "path/to/your/model.safetensors" ^
    --pruned-model subapps/layer_pruning/pruned_model_20layers.safetensors ^
    --samples subapps/layer_pruning/samples.json

echo Done!
pause
