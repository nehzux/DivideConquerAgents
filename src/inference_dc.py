# import keys and openai
from openai import OpenAI
import os
import tiktoken
import random
import sys
import math
import time
from multiprocessing import Pool
import torch.multiprocessing as mp
from functools import partial
import traceback

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


def split_str(s, n, overlap=0):
    chunk_size = len(s) // n
    remainder = len(s) % n
    chunks = []
    
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(s[start:end])
        start = end - overlap
    
    return chunks

def query_llm(prompt, model, tokenizer, ctx, client=None, temperature=0, max_new_tokens=128):
    input_ids = tokenizer.encode(prompt, disallowed_special=())
    if len(input_ids) > ctx:
        input_ids = input_ids[:ctx//2] + input_ids[-ctx//2:] # middle chunking
        prompt = tokenizer.decode(input_ids)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error Occurs: \"{str(e)}\"\t\tRetry at trial {tries}...")
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return

def get_pred_multi(args):
    item, i, model, base_url, api_key, tokenizer, task, ctx = args
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # msg = msg_prepare(item, task)

        context = str(item['context'])
        input_ids = tokenizer.encode(context, disallowed_special=())
        prompt_len = len(input_ids)

        if task == 'qalb':
            worker_template = "Please read the following text and analyze the question. Return three sentences that are most helpful in answering the question. \n\n<text>$DOC$</text>\n\nQuestion: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$" 
            manager_template = """There is a list of candidates. Each candidate has been given one sequential part of reference materials in order and each provides relevent retrieved sentenses to help you answer the question: \n$Q$\n\nRemember most candidates do not have enough information to answer the question correctly or completely. You have been provided with the candidates' responses:\n$responses$\n\nMake your answer short and give the answer directly without any explanation. If your answer is NA or no idea, it must be wrong and you should think again. Format your response as follows: "The correct answer is (A or B or C or D)"."""
        elif task == "sum":
            worker_template = "Read the book paragraphs below and summarize. \n\n<text>$DOC$</text>\n\n"
            manager_template = "There is a list of candidates. Each candidate has been given one sequential part of a book in order and each summarizes their respective part. You have been provided with their responses. Your task is to synthesize a summarization. \n\nResponses from candidates: \n$responses$\n\n"
        elif task == "qaib":
            worker_template = "Read the book paragraphs below and analyze the question. Return three sentences that are most helpful in answering the question. \n\n<text>$DOC$</text>\n\nQuestion: $Q$"
            manager_template = "There is a list of candidates. Each candidate has been given one sequential part of a book in order and each provides relevent retrieved sentenses to help you answer the question $Q$. Remember most candidates do not have enough information to answer the question correctly or completely. You have been provided with their responses. Make your answer shorter than 10 words. If your answer is NA or no idea, it must be wrong and think again. Give the answer directly without any explanation. \n\nResponses from candidates: \n$responses$\n\n"
        elif task == "char":
            worker_template = "Below is part of a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\". Summarize the characters one by one that appear in the script. Then return three sentences that are most helpful in guessing who that character is.\n\nThe dialogue:\n\n---\n\n<text>$DOC$</text>\n\n---\n\nEnd of dialogue.\n\n"
            manager_template = "There is a list of candidates. Each candidate has been given one sequential part of a dialogue in order and each provides relevent sentenses to provide you more information about character \"$$MASK$$\". Remember most candidates do not have enough information to answer the question correctly or completely. You have been provided with their responses. Firstly, aggregate all the characters from the candidates. Then, your task is to guess which character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.\n\nResponses from candidates: \n$responses$\n\n"
        elif task == "kv":
            worker_template = "Extract the value corresponding to the specified key in the JSON object below.\n\n$DOC$\n\nQuestion: $Q$\n If you don't have the key, return 'NA'."
            manager_template = "There is a list of candidates. Each candidate has been given some KV pairs and each tries to answer the question $Q$. Remember only one candidate has correct information to answer the question. You have been provided with their responses. Your task is to answer the question based on their provided information. Make your answer shorter than 10 words. If your answer is NA or no idea, it must be wrong and think again. Give the answer directly without any explanation. \n\nResponses from candidates: \n$responses$\n\n"
        elif task == "math":
            worker_template = "<text>$DOC$</text>\n\nQuestion: $Q$\nGive the answer without explanation."
            manager_template = "There is a list of candidates. Each candidate has been given some numbers and each tries to answer the question $Q$. You have been provided with their responses. Your task is to answer the question based on their provided information. Make your answer shorter than 10 words. If your answer is NA or no idea, it must be wrong and think again. Give the answer directly without any explanation. \n\nResponses from candidates: \n$responses$\n\n"
        else:
            print(f"Task {task} is not implemented.")
            return None
        input_ids = tokenizer.encode(worker_template, disallowed_special=())
        worker_template_len = len(input_ids)

        # calculate how many agents are needed
        assert ctx > worker_template_len, "worker template is longer than ctx budget"
        n_agents =  math.ceil(prompt_len / (ctx-worker_template_len))
        print(f"prompt_len: {prompt_len: >8}, worker_template_len: {worker_template_len}, worker_ctx: {ctx}, n_agents: {n_agents}")
        ctx_splits = split_str(context.strip(), n_agents, overlap=2048)

        worker_output = []
        for i in range(n_agents):
            if task == 'qalb':
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i]).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
            elif task == "sum":
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i])
            elif task == "qaib":
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i]).replace('$Q$', item['input'].strip())
            elif task == "char":
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i])
            elif task == "kv":
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i]).replace('$Q$', item['input'].strip())
            elif task == "math":
                worker_prompt = worker_template.replace('$DOC$', ctx_splits[i]).replace('$Q$', item['question'].strip())
            else:
                print(f"Task {task} is not implemented.")
                return None
            response = query_llm(worker_prompt, model, tokenizer, ctx, client, temperature=0.0, max_new_tokens=2048)
            if response != "" and response != None: worker_output.append(response.strip())

        responses = "\n".join([f"#{i+1}: {response}" for i, response in enumerate(worker_output)])
        if task == 'qalb':
            manager_prompt = manager_template.replace('$Q$', item['question'].strip()).replace('$responses$', responses)
        elif task == "sum":
            manager_prompt = manager_template.replace('$responses$', responses)
        elif task == "qaib":
            manager_prompt = manager_template.replace('$Q$', item['input'].strip()).replace('$responses$', responses)
        elif task == "char":
            manager_prompt = manager_template.replace('$DOC$', ctx_splits[0]).replace('$responses$', responses)
        elif task == "kv":
            manager_prompt = manager_template.replace('$Q$', item['input'].strip()).replace('$responses$', responses)
        elif task == "math":
            manager_prompt = manager_template.replace('$Q$', item['question'].strip()).replace('$responses$', responses)
        else:
            print(f"Task {task} is not implemented.")
            return None
        # print(manager_prompt)
        output = query_llm(manager_prompt, model, tokenizer, ctx, client, temperature=0.0, max_new_tokens=128)
    except Exception as e:
        print(f"Process failed for item {i}: {str(e)}")
        traceback.print_exc()
    if output == '':
        return ''
    response = output.strip()
    item['pred'] = response
    item['context'] = context[:100]
    return item

def main(args, config, tokenizer):
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(
        args.save_dir, 
        args.task + '-' + args.model + f"-len{args.len}-ctx{args.ctx}.jsonl"
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
    # print(data[0].keys())

    tasks = []
    for i, item in enumerate(data):
        tasks.append((item, i, config['model2str'][args.model], URL, API_KEY, tokenizer, args.task, args.ctx))

    with Pool(args.n_proc) as pool:
        results = pool.map(get_pred_multi, tasks)

    fout = open(out_file, 'a', encoding='utf-8')
    args.fout = fout
    for r in results:
        fout.write(json.dumps(r) + "\n")
    fout.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--task",   type=str, required=True)
    parser.add_argument("--ctx",    type=int, required=True)
    parser.add_argument("--len",    type=int, default=120000)
    parser.add_argument("--save_dir", type=str, default="../outputs")
    parser.add_argument("--n_proc", type=int, default=16)
    args = parser.parse_args()

    main(args, config, tokenizer)