import json
import argparse

def evaluate(dev_gold, dev_predict):
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    lable = [1, 2, 3, 4]
    l = len(dev_predict)
    # l = 181
    print(l)
    f1_micro = f1_score(dev_gold[:l], dev_predict[:l], labels=lable, average="micro")
    f1_macro = f1_score(dev_gold[:l], dev_predict[:l], labels=lable, average="macro")
    f1_weighted = f1_score(dev_gold[:l], dev_predict[:l], labels=lable, average="weighted")
    print("result:", f1_micro, f1_macro, f1_weighted)
    conf = confusion_matrix(dev_gold[:l], dev_predict[:l])
    print(conf)
    return f1_micro, f1_macro, conf
    

def read_ans(dev_file):
    with open(dev_file, 'r') as fp:
        data = json.load(fp)
    gold = []
    for d in data:
        ans = d["answer"]
        if 'Yes' in ans:
            gold.append(1)
        elif 'No' in ans:
            gold.append(2)
        elif 'Irrelevant' in ans:
            gold.append(3)
        else:
            gold.append(4)
    return gold


def read_pre(predict_file):
    data = []
    with open(predict_file, 'r') as fp:
        for line in fp:
            # print(line)
            line = line.strip('\n')
            data.append(line)
        # data = fp.load(fp)
    predict = []
    for d in data:
        idx = d.find('/INST]')
        # idx = d.find("/INST")
        # idx = 6
        prediction = d[idx:]  
        if 'Yes' in prediction:
            predict.append(1)
        elif 'No'in prediction:
            predict.append(2)
        elif 'Irrelevant' in prediction:
            predict.append(3)
        else:
            # print(prediction)
            predict.append(4)

    return predict


def main_eva(predict_file, metric_file):
    dev_file = "./sharc_raw/json/sharc_dev.json"
    with open(dev_file, 'r') as fp1:
        gold = json.load(fp1)
    pre = []
    with open(predict_file, 'r') as fp:
        for line in fp:
            # print(line)
            line = line.strip('\n')
            pre.append(line)
    # predict_file = args.predict_file
    print(predict_file)
    dev_gold = read_ans(dev_file)
    dev_predict = read_pre(predict_file)
    # differ = [[], [], [], []]
    # for i in range(len(dev_gold)):
    #     if dev_gold[i] != dev_predict[i]:
    #         if dev_gold[i] == 4:
    #             print(pre[i])
    #             exit()
    f1_micro, f1_macro, conf = evaluate(dev_gold, dev_predict)
    result = [{"f1_micro":f1_micro.tolist(),"f1_macro":f1_macro.tolist(), "conf":conf.tolist()}]
    with open(metric_file, 'w') as fp:
        fp.write(json.dumps(result, indent=2))
    fp.close()
    # return f1_micro

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # #type是要传入的参数的数据类型  help是该参数的提示信息
    # parser.add_argument('--predict_file', type=str)
    # args = parser.parse_args()
    predict_file = "/raid_sda/home/rjm/mine/mistral-add=True_cross=True-scenario=False-0528-0129-5epoch/checkpoint-10310/repeat_0_add&mlp_results.json"
    metric_file = "/raid_sda/home/rjm/mine/mistral-add=True_cross=True-scenario=False-0528-0129-5epoch/checkpoint-10310/present_metric_0.json"
    main_eva(predict_file, metric_file)
