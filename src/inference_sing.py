# import keys and openai
from openai import OpenAI
import os
import tiktoken
import random
import sys
from multiprocessing import Pool
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def tokenized_length(text):
    return len(tokenizer.encode(text))

import json
with open('config.json', 'r') as f:
    config = json.load(f)
keys = config['keys']
TOGETHER_API_KEY = keys['TOGETHER_API_KEY']
OPENAI_API_KEY = keys['OPENAI_API_KEY']

LOCAL_URL = "http://127.0.0.1:8000/v1"
TOGETHER_URL = "https://api.together.xyz/v1"
OPENAI_URL = "https://api.openai.com/v1"


def msg_prepare(item, task):
    if task == 'math':
        msg = str(item['context']) + '\n' + item['question'] + '\nGive the answer without explanation'
    elif task == 'kv':
        msg = str(item['context']) + '\n' + item['input'] + '\nGive the answer without explanation'
    elif task == 'qalb':
        template_0shot = open('0shot.txt', encoding='utf-8').read()
        msg = template_0shot.replace('$DOC$', item['context'].strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
    elif task == 'sum':
        template = "Read the book paragraphs below and summarize. \n\nBook: {context}\n"
        msg = template.format(context=item['context'])
    elif task == 'char':
        template = "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else."
        msg = template.format(context=item['context'])
    elif task == 'qaib':
        template = "Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise."
        msg = template.format(context=item['context'], question=item['input'])
    else:
        pass
    return msg

def process_item(args):
    item, i, out_file, model, base_url, api_key, tokenizer, task = args

    try:
        # Create a local OpenAI client inside each worker
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        msg = msg_prepare(item, task)
        length = len(tokenizer.encode(msg))
        # print(f"Input message length: {length}")

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                max_tokens=config['task2maxlen'][task],
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed for item {i}: {str(e)}")
            response = "ERROR"

        with open(out_file, 'a') as f:
            f.write(json.dumps({
                "id": i,
                "answer": item["answer"],
                "pred": response
            }) + "\n")
    except Exception as e:
        print(f"Process failed for item {i}: {str(e)}")

def main(args, config, tokenizer):
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(
        args.save_dir, 
        args.task + '-' + args.model + f"-len{args.len}.jsonl"
    )
    print(out_file)

    if 'gpt' in args.model:
        URL = OPENAI_URL
        API_KEY = OPENAI_API_KEY
    else:
        URL = TOGETHER_URL
        API_KEY = TOGETHER_API_KEY

    task_file = f'../data/{args.task}_{args.len}.jsonl'
    with open(task_file, 'r') as f:
        data = [json.loads(line) for line in f]
    print(data[0].keys())

    # Prepare arguments for each item
    tasks = []
    for i, item in enumerate(data):
        tasks.append((item, i, out_file, config['model2str'][args.model], URL, API_KEY, tokenizer, args.task))

    with Pool(args.n_proc) as pool:
        pool.map(process_item, tasks)

# argparse in main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--task",   type=str, required=True)
    parser.add_argument("--len",    type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../outputs")
    parser.add_argument("--n_proc", type=int, default=16)
    args = parser.parse_args()

    main(args, config, tokenizer)


