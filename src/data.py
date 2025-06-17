import json
import tiktoken
import random
import jsonlines
import sys
import numpy as np

tokenizer = tiktoken.encoding_for_model("gpt-4o")

def tokenized_length(text):
    return len(tokenizer.encode(text))

def gen_math_data(len=120, samples=100):
    data = []
    question_types = [
        "What is the largest number?",
        "What is the 2nd largest number?",
        "What is the smallest number?",
        "What is the 2nd smallest number?"
    ]
    for i in range(samples):
        # context = [random.randint(1, 100) for _ in range(len // 3)]
        context = [int(random.gauss(50, 15)) for _ in range(len // 3)]
        qtype = question_types[i % 4]
        sorted_nums = sorted(context)
        if "largest" in qtype and "2nd" not in qtype:
            ans = sorted_nums[-1]
        elif "2nd largest" in qtype:
            ans = sorted_nums[-2]
        elif "smallest" in qtype and "2nd" not in qtype:
            ans = sorted_nums[0]
        elif "2nd smallest" in qtype:
            ans = sorted_nums[1]
        data.append({
            "id": i,
            "context": context,
            "question": qtype,
            "answer": ans
        })
    return data

def gen_kv_data():

    len_list = np.array([1000, 2000, 4000, 8000, 15000, 30000, 60000, 120000])
    nsample = [100] * len(len_list)
    nnoise = len_list / 50

    for ii in range(len(len_list)):
        print(len_list[ii], nsample[ii], nnoise[ii])
        cnt = -1
        ret = []

        with jsonlines.open("../data/kv-retrieval-3000_keys.jsonl") as fin:
            for line in fin:
                # print(len(line["ordered_kv_records"]))
                # print(tokenized_length(str(line["ordered_kv_records"][0]))) 50
                # print(line["ordered_kv_records"][0])  ['0198fbb7-e2d6-4099-89ac-d1a1645772da', '99c9a6b5-3328-4ab0-93cd-41796ff990d2']
                cnt += 1
                if cnt == nsample[ii]:
                    break
                ans_id = min(int(cnt * nnoise[ii] / nsample[ii]), nnoise[ii])

                text = "JSON data:\n{"
                t = -1
                random.shuffle(line["ordered_kv_records"])
                for item in line["ordered_kv_records"]:
                    t += 1
                    if t == nnoise[ii]:
                        break
                    text += "\"" + item[0] + "\": \"" + item[1] + "\", "
                text = text[:-2] + '}'
                question = "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                ret.append({"id": cnt, "context": text, "input": question, "answer": line["ordered_kv_records"][ans_id][1]})
            
        
        fw = jsonlines.open(f"kv_{len_list[ii]}.jsonl", 'w')
        fw.write_all(ret)
        fw.close()

def prepare_sum(len_min=100000, len_max=120000):
    filtered_data = []
    with jsonlines.open("../data/longbook_sum_eng.jsonl") as fin:
        for line in fin:
            context_length = tokenized_length(line["context"])
            if len_min <= context_length <= len_max:
                filtered_data.append(line)
    
    with jsonlines.open(f"sum_{len_max}.jsonl", 'w') as fout:
        fout.write_all(filtered_data)

def prepare_char(len_min=100000, len_max=120000):
    filtered_data = []
    with jsonlines.open("../data/longdialogue_qa_eng.jsonl") as fin:
        for line in fin:
            context_length = tokenized_length(line["context"])
            if len_min <= context_length <= len_max:
                filtered_data.append(line)
    
    with jsonlines.open(f"char_{len_max}.jsonl", 'w') as fout:
        fout.write_all(filtered_data)

def prepare_qa(len_min=100000, len_max=120000):
    filtered_data = []
    with jsonlines.open("../data/longbook_qa_eng.jsonl") as fin:
        for line in fin:
            context_length = tokenized_length(line["context"])
            if len_min <= context_length <= len_max:
                filtered_data.append(line)
    
    with jsonlines.open(f"qaib_{len_max}.jsonl", 'w') as fout:
        fout.write_all(filtered_data)

def main():
    # with open("../data/math_find.jsonl", "r") as f:
    #     items = [json.loads(line) for line in f]
    # first_item = items[3]
    # print(first_item.keys())
    # parsed_context = json.loads(first_item["context"])
    # print(len(parsed_context))
    # context_length = tokenized_length(first_item["context"])
    # print(context_length)
    for task_len in [1000, 2000, 4000, 8000, 15000, 30000, 60000, 120000]:
        data = gen_math_data(len=task_len)
        with open(f"../data/math_{task_len}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

def json_to_jsonl(json_file, jsonl_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    with open(jsonl_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    # main()
    # gen_kv_data()
    # with jsonlines.open("./kv_8000.jsonl") as f:
    #     for line in f:
    #         context_dict = json.loads(line['context'][11:])
    #         print(len(context_dict))
    #         key_id = list(context_dict.keys()).index(line['input'].split('"')[1])
    #         print(f"ID: {line['id']}, Context Length: {tokenized_length(json.dumps(context_dict))}, Key ID: {key_id}")
    # json_to_jsonl('../data/qalb_120000.json', '../data/qalb_120000.jsonl')
    # prepare_sum()
    # prepare_char()
    prepare_qa()
    