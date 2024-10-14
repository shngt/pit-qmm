import os
import json

# Path to the JSON file
json_file = '/proj/esv-summer-interns/home/eguhpas/pit-qmm/playground/data/ft/point/train_split_1.json'

# Read the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Filter JSON entries based on existence of corresponding .npy file
filtered_data = []
for entry in data:
    if 'ColorNoise' in entry['image']:

# Write filtered data to a new JSON file
filtered_json_file = '/proj/esv-summer-interns/home/eguhpas/PointLLM/data/anno_data/filtered_lspcqa_instruct_discrete.json'
with open(filtered_json_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"Filtered JSON file has been saved to {filtered_json_file}.")
