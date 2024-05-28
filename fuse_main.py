from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import os 
import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

import time
from evaluate import main_eva
from tqdm import tqdm

from evaluate import main_eva, read_ans, evaluate
from utils import print_trainable_parameters
from dataset import load_dataset
from finetuning_args import ScriptArguments

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                # unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
                # unwrapped_model = unwrapped_model.merge_and_unload()        
                # save_dir = f"{args.output_dir}/chechkpoint-{state.global_step + 1}/merged/"   
                unwrapped_model.save_pretrained(
                    args.output_dir, state_dict=state_dict
                )
                # self.trainer.tokenizer.save_pretrained(args.output_dir)
                # model.config.to_json_file("save_config.json")
            self.trainer.accelerator.wait_for_everyone()
        return control



def main():
    def read_pre(decoded_preds):
        predict = []
        for prediction in decoded_preds:
            if '/INST]' in prediction:
                prediction = str(prediction[prediction.find('/INST]'):])
                if 'yes' in prediction.lower():
                    predict.append(1)
                elif 'no'in prediction.lower():
                    predict.append(2)
                elif 'irrelevant' in prediction.lower():
                    predict.append(3)
                else:
                    predict.append(4)
            else:
                predict.append(3)
        return predict
        
    def compute_metrics(pred):
        preds, labels = pred
        preds = np.argmax(preds, axis=-1)
        # print(preds)
        # exit()
        # print(labels)

        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # for ind1 in labels:
        #     for ind2 in ind1:
        #         if ind2 == -100:
        #             ind2 = tokenizer.pad_token_id
        # print(labels)
        # decoded_preds = tokenizer.batch_decode(preds[0], skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        print(decoded_labels)

        predictions = read_pre(decoded_labels)
        # print(predictions)
        dev_file = "./sharc_raw/json/sharc_train_question_fixed_eval.json"
        goldens = read_ans(dev_file)
        # print(goldens)
        f1_micro, f1_macro, conf = evaluate(goldens, predictions)
        result = {"f1_micro":f1_micro.tolist(),"f1_macro":f1_macro.tolist(), "conf":conf.tolist()}
        #print(result)
        #exit()
        return result

    # from main import ScriptArguments
    from transformers import HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments
    parser = HfArgumentParser([ScriptArguments, Seq2SeqTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    from typing import cast
    args = cast(ScriptArguments, script_args)
    training_args = cast(Seq2SeqTrainingArguments, training_args)
    print(args.fuse)
    from transformers import AutoTokenizer
    if args.load_checkpoint:    
        model_path = args.checkpoint_path
    else:
        model_path = "/raid_sda/home/rjm/mine/Mistral-7B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # tokenizer.pad_token = "!"
    
    # TRAIN_DATA

    # train_file = "./data/sharc_raw/json/sharc_train.json"
    # train_file = "./data/sharc_raw/json/sharc_train_question_fixed.json"
    # cut dev data from train
    train_file = "./sharc_raw/json/sharc_train_question_fixed_splited.json"
    # train_file = "./sharc_raw/json/train_test.json"
    # train_doc_file = "./data/train_snippet_parsed.json"
    
    train_doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"
    train_sce_file = "./RotatE_sharc_Sce/parsed/train_sce_parsed_with_id.json"
    # train_data = load_dataset(args, train_file, train_doc_file, tokenizer, train_sce_file)

    # EVAL_DATA
    dev_file = "./sharc_raw/json/sharc_dev.json"
    # dev_file = "./data/sharc_raw/json/sharc_dev_question_fixed.json"
    # # dev_doc_file = "./data/dev_snippet_parsed.json"
    dev_doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
    dev_sce_file = "./RotatE_sharc_Sce/parsed/dev_sce_parsed_with_id.json"
    eval_data = load_dataset(args, dev_file, dev_doc_file, tokenizer, dev_sce_file)
    # eval_file = "./sharc_raw/json/sharc_train_question_fixed_eval.json"
    # dev_doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"
    # dev_sce_file = "./RotatE_sharc_Sce/parsed/train_sce_parsed_with_id.json"
    # eval_data = load_dataset(args, eval_file, dev_doc_file, tokenizer, dev_sce_file)

    # TEST_DATA
    # dev_file = "./data/sharc_raw/json/sharc_dev.json"
    # # dev_file = "./data/sharc_raw/json/sharc_dev_question_fixed.json"
    # # dev_doc_file = "./data/dev_snippet_parsed.json"
    # dev_doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
    # dev_data = load_dataset(args, dev_file, dev_doc_file, tokenizer)

    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
    from modeling_mistral import MistralForCausalLM
    import torch
    # model = MistralForCausalLM.from_pretrained("/home/rjm/mine/Mistral-7B-instruct", torch_dtype=torch.float16)
    # model = MistralForCausalLM.from_pretrained("/home/rjm/mine/mistral-fuse-add-4epoch-qv-lora-save/checkpoint-100", torch_dtype=torch.float16)
    # model = MistralForCausalLM.from_pretrained("/home/rjm/mine/Mistral-7B-instruct") 
    model =  MistralForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    for name, param in  model.named_parameters():
        if "add" in name: 
            # print(name, param.requires_grad)
            param.requires_grad = True
        #     print(name, param.requires_grad)
        if "edu_project" in name:
            param.requires_grad = True
        if "cross_attn" in name:
            param.requires_grad = True
            
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # target_modules=[ "w1", "w2", "w3"],  #Only Training the "expert" layers
        target_modules=[
            "q_proj",  
            "v_proj",
            "up_proj",
            "down_proj",
        ],
        # target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["edu_embed", "edu_project_down", "edu_project_up", "cross_attn"]
    )
    # model = get_peft_model(model, config)
    # print(next(model.parameters()).device)
    # model.to("cuda")
    # for name, param in  model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    

    # for name, param in  model.named_parameters():
    #     print(name, param.requires_grad)
    
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        # train_dataset=train_data,
        eval_dataset=eval_data[:120],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    # Create trainer and add callbacks
    model.config.use_cache = False
    trainer.accelerator.print(f"{trainer.model}")
            
    # trainer.model.print_trainable_parameters()
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    
    # if args.load_checkpoint:
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    trainer.evaluate()
    # exit()

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    # old_ans = 0
    # model.eval()
    # with torch.no_grad():
    #     ans = evaluate(args, model, tokenizer, dev_data)
    # if ans > old_ans:
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    # old_ans = ans
    trainer.accelerator.wait_for_everyone()

    # merge adapters
    # Save everything else on main process
    if trainer.args.process_index == 0:
        if script_args.merge_adapters:
            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            from peft import AutoPeftModelForCausalLM

            # load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )  
            # Merge LoRA and base model and save
            model = model.merge_and_unload()        
            model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="8GB"
            )
        else:
            trainer.model.save_pretrained(
                training_args.output_dir, safe_serialization=True
            )

        # save tokenizer 
        tokenizer.save_pretrained(training_args.output_dir)

    

def predict(model_path, count):
    from transformers import HfArgumentParser, TrainingArguments
    parser = HfArgumentParser([ScriptArguments,TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    from typing import cast
    args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    print(args.test, args.fuse)
    # generation_config = dict(
    #     # temperature=0.4,
    #     # top_p=0.1,
    #     # top_k=30,
    #     # top_p=0.9,
    #     # do_sample=False,
    #     # num_beams=1,
    #     # repetition_penalty=1.3,
    #     max_new_tokens=400
    # )

    # from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import deepspeed
    import torch
    from modeling_mistral import MistralForCausalLM
    print("load")
    # model_path = "/home/rjm/mine/mistral-cross-0509-2343-4epoch-qv-lora-save/checkpoint-2000"
    
    # tokenizer_path = "/home/rjm/mine/data/checkpoint-2500"
    tokenizer_path = "/raid_sda/home/rjm/mine/Mistral-7B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    dev_file = "./sharc_raw/json/sharc_dev.json"
    # dev_file = "./data/sharc_raw/json/sharc_dev_question_fixed.json"
    # # dev_doc_file = "./data/dev_snippet_parsed.json"
    dev_doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
    dev_sce_file = "./RotatE_sharc_Sce/parsed/dev_sce_parsed_with_id.json"
    dev_data = load_dataset(args, dev_file, dev_doc_file, tokenizer, dev_sce_file)
    # print(dev_data[0])

    # eval_file = "/home/rjm/mine/data/sharc_raw/json/sharc_train_question_fixed_eval.json"
    # dev_doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"
    # dev_sce_file = "./RotatE_sharc_Sce/parsed/train_sce_parsed_with_id.json"
    # dev_data = load_dataset(args, dev_file, dev_doc_file, tokenizer, dev_sce_file)
    
    from peft import load_peft_weights, set_peft_model_state_dict
    base_model = "/raid_sda/home/rjm/mine/Mistral-7B-instruct"
    model = MistralForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model.load_lora_weights(model_path)
    model.fuse_lora()
    # adpter_weights = load_peft_weights(model_path)
    # set_peft_model_state_dict(model, adpter_weights)
    # edu_embed_matrix = np.load("/home/rjm/mine/RotatE_sharc_4/all_entity_embedding.npy")
    # _, dim = edu_embed_matrix.shape
    # edu_zero = np.zeros([1, dim])
    # edu_embed_matrix = torch.FloatTensor(np.concatenate([edu_zero, edu_embed_matrix], axis=0))
    # model.model.edu_embed = torch.nn.Embedding.from_pretrained(edu_embed_matrix)
    # print(next(model.parameters()).device)
    model.to("cuda")
    # ds_model = deepspeed.init_inference(
    #     model=model,      # Transformers模型
    #     mp_size=2,        # GPU数量
    #     dtype=torch.float16, # 权重类型(fp16)
    #     replace_method="auto", # 让DS自动替换层
    #     replace_with_kernel_inject=True, # 使用kernel injector替换
    # )
    # print(f"模型加载至设备{model.module.device}\n")
    # from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
    # assert isinstance(ds_model.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"

    
    # def gen_after_tunning(model, dev_data, maxlen=1024, sample=True):
    results = []
    model.eval()
    # model.eval()
    # count = 0
    with torch.no_grad():
        for data in tqdm(dev_data):
            # out = model(**data)
            # print(out, out.keys())
            output = model.generate(
                    **data, 
                    max_new_tokens=256, 
                    num_return_sequences=1,
                    # temperature=0.0
                    # top_p=0.1,
                    # **generation_config
                    )
            # output = out["logits"].detach().cpu().numpy()
            # output = np.argmax(output, axis=-1)  
            res = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(res)
            # exit()
            save_file = f"{model_path}/repeat_{count}_add&mlp_results.json"
            import json
            res = res.replace("\n", "\\n")
            results.append(res)
            with open(save_file, 'a') as fp:
                fp.write(res + '\n')
    metric_file = f"{model_path}/repeat_{count}_metric.json"
    main_eva(save_file, metric_file)  



def main_predict():
    for root, dirs, files in os.walk("/home/rjm/mine/mistral-cross-0509-2343-4epoch-qv-lora-save"):
        for dir in dirs:
            if "checkpoint" in dir and 'tmp' not in dir:
                file_path = os.path.join(root, dir)
                print(file_path)
                # exit()
                predict(file_path)
                

if __name__=='__main__':
    # train_file = "./data/sharc_raw/json/sharc_train_question_fixed.json"
    # split_train(train_file)
    # main()
    # read_Graph_embedding()
    # entity_dict = read_entity()
    # change_doc_file(entity_dict)

    # main_predict()
    # path = "/home/rjm/mine/mistral-cross-0509-2343-4epoch-qv-lora-save/checkpoint-1650"
    # for i in range(4):
    #     predict(path, i)
    path = "/raid_sda/home/rjm/mine/mistral-add=True_cross=True-scenario=False-0528-0129-5epoch/checkpoint-10310"
    count = 1
    predict(path, 1)
    # entity_dict = read_entity()
    # change_sce_file(entity_dict)