from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

import torch
from peft import LoraConfig, TaskType, get_peft_model

system_prompt="你擅长文本分类，能判断一段文字描述来自human还是chatgpt。你只需要回答一个单词，回答human或者chatgpt。"

def creat_input(question,answer):
    length=len(question)
    return str("这里有一段关于"+question[24:length]+"的文字描述，请判断它来自human还是chatgpt：\n"+answer)

def change_(question,answer,type):
    MAX_LENGTH = 512

    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer("\n".join(["<|im_start|>system",system_prompt+"<|im_end|>"+"\n<|im_start|>user\n"+creat_input(question,answer)+"<|im_end|>\n"]).strip(), add_special_tokens=False)  # add_special_tokens：是否在开头加 special_tokens
    response = tokenizer("<|im_start|>assistant\n"+type+"<|im_end|>\n",add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen的特殊构造就是这样的

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
# 用于处理数据集的函数
def change(dataset):
    ds=[]
    cnt=dataset.num_rows
    for i in range(cnt):
        example=dataset[i]
        ds.append(change_(example["question"],example["human_answers"][0],"human"))
        ds.append(change_(example["question"],example["chatgpt_answers"][0],"chatgpt"))
    return Dataset.from_list(ds)

def chat(model, tok, query, system=[]):
    # print("【type(system)】",type(system))
    # print("【type(query)】",type(query))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query}
    ]
    text = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=1)
    model_inputs = tokenizer([text],return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

import modelscope
from modelscope.msdatasets import MsDataset

# 对指定数据调用模型进行预测
def pred(model,tokenizer,data,print_info=10): #每print_info个数据输出一次信息
    model.eval()
    acc,tot=0,0
    data_len=len(data)
    # print("pred_len=",data_len)
    data_len=30
    for i in range(data_len):
        example=data[i]
        test_input_1=creat_input(example["question"],example["human_answers"][0])
        response_1=chat(model,tokenizer,test_input_1,system=system_prompt)
        acc+=int(response_1=="human" or response_1=="人类")
        test_input_2=creat_input(example["question"],example["chatgpt_answers"][0])
        response_2=chat(model,tokenizer,test_input_2,system=system_prompt)
        acc+=int((response_2=="chatgpt"))
        # print("【system】",system_prompt)
        # print("【test_input】",test_input)
        # print("【response】",response)
        tot+=2
        # print("Test No.",i)
        if (print_info>0 and i%print_info==0):
            print("Test NO.{}: response=[{},{}] acc=({}/{})".format(i,response_1,response_2,acc,tot))
        if i>50:
            break
    return acc/tot

if "__main__" == __name__:
  
    # ds=MsDataset.load('simpleai/HC3-Chinese', subset_name='baike', split='train') #调用数据集
    #数据集大小: 4617
    #每个数据形如 { id, question, human_answers, chatgpt_answers}，其中answers为list(大小都为1)
    # print("【ds[0]】：")
    # print(ds[0]['id'])
    # print(ds[0]['question'])
    # print(ds[0]['human_answers'][0])
    # print(ds[0]['chatgpt_answers'][0])

    # str_=ds[0]['question']
    # length=len(str_)
    # print(str_[24:length])

    #for i in range(len(ds[0]['question'])):
    #    print("({}) [ {} ]".format(i,ds[0]['question'][i]))

    # 处理数据集
    HC3=MsDataset.load('simpleai/HC3-Chinese', subset_name='baike', split='train') #调用HC3数据集
    dataset=HC3.to_hf_dataset() #转换成huggingface dataset
    data_=dataset.train_test_split(train_size=0.8,seed=233) #数据集划分
    # data_["train"], data_["validation"], data_["test"]
    print("【数据集加载完成】")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 以半精度形式加载模型
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
    print("【模型加载完成】")

    # 将数据集改成token形式
    data_token = change(data_["train"])
    print("【训练集转token完成】")
    print("训练集大小: 2*{}={}".format(len(data_["train"]),len(data_token)))

    # 配置LoRA参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proh", "v_proj"],  # 这个不同的模型需要设置不同的参数，需要看模型中的attention层
        inference_mode=False, # 关闭推理模式 (即开启训练模式)
        r=8, # Lora 秩
        lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1 # Dropout 比例
    )
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir="./output/Qwen",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=1, #多少步输出一次log
        num_train_epochs=3,
        gradient_checkpointing=True,
        save_steps=100, #多少步保存一次模型
        learning_rate=1e-4,
        save_on_each_node=True,
        # prediction_loss_only=True
    )

    print("测试集大小：",len(data_["test"]))

    print("Test准确率：acc={}",pred(model,tokenizer,data_["test"],1))
