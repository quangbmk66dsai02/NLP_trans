import json

def extract_from_jsonl(file_path):
    extracted_data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            extracted_data.append(json_obj)
    return extracted_data

# Example usage:
jsonl_file_path = 'train.jsonl'
data = extract_from_jsonl(jsonl_file_path)
print(data)