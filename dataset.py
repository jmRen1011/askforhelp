import torch

def load_dataset(args, train_file, doc_file, tokenizer, sce_file):
    import json
    def read_data(file):
        with open(file, 'r') as fp:
            data = json.load(fp)
        return data  

    def generate_prompt(data, doc_data):
        # 直接拼成[INST]的形式 也可以
        # sys_msg= "Translate the given text to Shakespearean style."
        sys_msg = "You are a helpful bot who reads texts and answers questions about them."
        # print(data)
        tree_id = data["tree_id"]
        cur_doc = doc_data[tree_id]
        # cur_sce = sce_data[tree_id]["s_clauses"]
        #print(cur_sce)
        # exit()
        # print(cur_doc)
        # if cur_doc 
        edu_text = ''
        for edus in (cur_doc["edus"] if not cur_doc["is_bullet"] else cur_doc["clauses"]):
            if type(edus) == list:
                for edu in edus:
                    # if edu[0] ==
                    if edu_text == '':
                        edu_text = edu
                        continue
                    edu_text = edu_text + '\t' + edu
            else:
                if edu_text == '':
                    edu_text = edus
                    continue
                edu_text = edu_text + '\t' + edus
        # print(edu_text.split("\t"))
        # exit()    
        history = ''
        for his in data["history"]:
            # print(his)
            if not history:
                history = his["follow_up_question"] + ' ' + his["follow_up_answer"]
            history =  history + '\t' + his["follow_up_question"] + ' ' + his["follow_up_answer"]
        if len(history) == 0:
            history = 'Empty'
        scenario = data["scenario"]
        if len(scenario) == 0:
            scenario = 'Empty'
        question = data["question"]
        answer = data["answer"]
        if args.test:
            if args.fuse:
                prompt = ("[INST]" + sys_msg + " Document:", 
                      edu_text.split("\t"), 
                      " Scenario:" + scenario + " Dialogue History:" + history + " Question:" + question + "[/INST]" + " Answer:")
                # print(prompt)
            else:
                prompt = "[INST]" + sys_msg + "Document:" + edu_text + " Scenario:" + scenario + " Dialogue History:" + history + " Question:" + question + "[/INST] " + " Answer:"
        elif args.fuse and not args.test:
            prompt = ("[INST]" + sys_msg + " Document:", 
                      edu_text.split("\t"), 
                      " Scenario:" + scenario + " Dialogue History:" + history + " Question:" + question + "[/INST]" + " Answer:" + answer)
        else:
            prompt = "[INST]" + sys_msg + "Document: " + edu_text + "Scenario: " + scenario + "Dialogue History: " + history + "Question: " + question + "[/INST] " + " Answer:" + answer
        # print(prompt)
        # exit()
        return prompt
    
    def tokenize(args, tokenizer, prompt):
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=args.cutoff_len ,
            padding="max_length"
        )
    
    def tokenize_end(tokenizer, prompt):
        return tokenizer(
            prompt + tokenizer.eos_token,
            return_tensors="pt"
        ).to("cuda")
    
    def tokenize_test(args, tokenizer, prompt):
        return tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len ,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
    
    def tokenize_plt(args, tokenizer, prompt):
        return tokenizer(prompt)
    
    def tokenize_token(tokenizer, token):
        return tokenizer(token, return_tensors="pt").to("cuda")
    
    doc_data = read_data(doc_file)
    # sce_data = read_data(sce_file)
    train_ori_data = read_data(train_file)
    from tqdm import tqdm 
    if args.test and not args.fuse:
        print("using test")
        train_data = [tokenize_test(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    elif args.plt:
        train_data = [tokenize_plt(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    elif args.fuse:
        print("using fuse")
        train_data = []
        for x in tqdm(train_ori_data):
            # print(x)
            prompt = generate_prompt(x, doc_data)
            # print(prompt)

            pre_ids = tokenize_token(tokenizer, prompt[0])
            input_ids = pre_ids["input_ids"]
            attention_mask = pre_ids["attention_mask"]
            edu_ids = torch.full(pre_ids["input_ids"].shape, 0, dtype=torch.int)
            # print("0:", input_ids.shape, edu_ids.shape)
            # edu_ids = set()

            # [1, 22, 4096]
            edu_indices = doc_data[x["tree_id"]]["enity_ids"]
            # sce_indices = sce_data[x["tree_id"]]["enity_ids"]
            # print(prompt[1],edu_indices)
            for idx, edu in enumerate(prompt[1]):
                # print(edu)
                edu_input_ids = tokenize_token(tokenizer, edu)
                edu_index = int(edu_indices[idx])
                input_ids = torch.cat(
                    [input_ids, edu_input_ids["input_ids"][0][1:].unsqueeze(0)], dim=1
                )
                attention_mask = torch.cat(
                    [attention_mask, edu_input_ids["attention_mask"][0][1:].unsqueeze(0)], dim=1
                )
                # print(edu_input_ids["input_ids"][0][1:].unsqueeze(0).shape)
                s = edu_input_ids["input_ids"][0][1:].unsqueeze(0).shape
                # print(s)
                to_cat = torch.full(tuple(s), edu_index, dtype=torch.int)
                # print(to_cat.shape)
                edu_ids = torch.cat(
                    [edu_ids, to_cat], dim=1
                )
                # print(edu_ids.shape)
                # edu_ids.add(edu_index)
                
            
            if args.test:
                post_ids = tokenize_token(tokenizer, prompt[-1])
            else:
                post_ids = tokenize_end(tokenizer, prompt[-1])
            
            input_ids = torch.cat(
                [input_ids, post_ids["input_ids"][0][1:].unsqueeze(0)], dim=1
            )
            attention_mask = torch.cat(
                [attention_mask, post_ids["attention_mask"][0][1:].unsqueeze(0)], dim=1
            )
            s_1 = post_ids["input_ids"][0][1:].unsqueeze(0).shape
            to_cat_1 = torch.full(tuple(s_1), 0, dtype=torch.int)
            edu_ids = torch.cat(
                [edu_ids, to_cat_1], dim=1
            )
            # print("2:",  input_ids.shape, edu_ids.shape)
            # exit()
            # print(input_ids)
            # print(tokenizer.batch_decode(input_ids[0]))
            # exit()

            # if not edu_ids:
            #     edu_ids.add(0)
            # edu_ids = torch.LongTensor(list(edu_ids))
            # print(input_ids.shape, attention_mask.shape, edu_ids.shape)
           
            # print(input_ids, attention_mask, edu_ids)
            # print(input_ids.shape, attention_mask.shape, edu_ids.shape)

            # print(input_ids)
            # print(edu_ids)

            if args.test:
                if args.fuse:
                    # print("here")
                    train_data.append({
                        "input_ids": input_ids.to("cuda"),
                        "attention_mask": attention_mask.to("cuda"),
                        "edu_ids": edu_ids.to("cuda")
                    })
                else:
                    train_data.append({
                        "input_ids": input_ids[0],
                        "attention_mask": attention_mask[0]
                    })   
            else:
                train_data.append({
                    "input_ids": input_ids[0].to("cuda"),
                    "attention_mask": attention_mask[0].to("cuda"),
                    "edu_ids": edu_ids[0].to("cuda")
                })
            
    else:
        train_data = [tokenize(args, tokenizer, generate_prompt(x, doc_data)) for x in tqdm(train_ori_data)]
    # print(train_data[0])
    # exit()
    return train_data