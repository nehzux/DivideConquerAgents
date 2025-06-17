import json
import sys
import csv
import argparse
import re
import ast
import glob
import os
from compute_scores import (
    get_score_one_kv_retrieval,
    get_score_one_longbook_sum_eng,
    get_score_one_longdialogue_qa_eng,
    get_score_one_longbook_qa_eng
)
with open('config.json', 'r') as f:
    config = json.load(f)

len2real = config['len2real']

def extract_answer_qalb(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None
        
def compute_accuracy(jsonl_file, task):
    if task == 'math':
        correct, total = 0, 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = str(record.get('answer'))
                pred = record.get('pred')
                if ans == pred:
                    correct += 1
                total += 1
        return correct / total if total else 0.0
    elif task == 'kv':
        correct, total = 0, 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = str(record.get('answer'))
                pred = record.get('pred')
                correct += get_score_one_kv_retrieval(pred, ans, None)
                total += 1
        return correct / total if total else 0.0
    elif task == 'qalb':
        correct, total = 0, 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = str(record.get('answer'))
                pred = extract_answer_qalb(record.get('pred'))
                if ans == pred:
                    correct += 1
                total += 1
        return correct / total if total else 0.0
    elif task == 'sum':
        score = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = str(record.get('answer'))
                pred = record.get('pred')
                score.append(get_score_one_longbook_sum_eng(pred, ans, None))
        return sum(score) / len(score) if score else 0.0
    elif task == 'char':
        correct, total = 0, 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = ast.literal_eval(str(record.get('answer')))
                pred = record.get('pred')
                correct += get_score_one_longdialogue_qa_eng(pred, ans, None)
                total += 1
        return correct / total if total else 0.0
    elif task == "qaib":
        score = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                ans = ast.literal_eval(str(record.get('answer')))
                pred = record.get('pred')
                score.append(get_score_one_longbook_qa_eng(pred, ans, None))
        return sum(score) / len(score) if score else 0.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute accuracy for different task lengths.')
    parser.add_argument('--model', type=str, help='The model name to use in the output file paths.')
    parser.add_argument('--task', type=str, help='The task name to use in the output file paths.')
    args = parser.parse_args()

    model = args.model

    # Glob files matching the pattern
    pattern = f'../outputs/{args.task}-{args.model}-*.jsonl'
    files = glob.glob(pattern)
    
    # Extract lengths and sort files by length
    file_length_pairs = []
    for file in files:
        filename = os.path.basename(file)
        # Extract length from filename like "task-model-lenXXX.jsonl"
        if filename.startswith(f'{args.task}-{args.model}-len') and filename.endswith('.jsonl'):
            length_str = filename[len(f'{args.task}-{args.model}-len'):-len('.jsonl')]
            try:
                length = int(length_str)
                file_length_pairs.append((file, length))
            except ValueError:
                print(f"Warning: Could not extract length from {filename}")
    
    # Sort by length
    file_length_pairs.sort(key=lambda x: x[1])
    
    if not file_length_pairs:
        print(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    csv_file = f'../outputs/results-{args.task}-{args.model}.csv'
    fieldnames = ['task', 'task_length', 'model', 'accuracy']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for output_file, task_len in file_length_pairs:
            parts = os.path.basename(output_file).split('-')
            task = parts[0]
            model = parts[1]
            real_task_len = len2real[str(task_len)]
            accuracy = compute_accuracy(output_file, task=args.task)
            writer.writerow({'task': task, 'task_length': real_task_len, 'model': model, 'accuracy': accuracy})
            print(f"{output_file}, Accuracy: {accuracy}")