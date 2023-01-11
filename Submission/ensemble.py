import ast
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from argparse import ArgumentParser

def majority(pixel_predictions : list, weight):
    '''
    다수결
    주의점: 만약 같은 개수면 class번호가 낮은 것으로 결정됨
    작동방식
        - 입력: 한 픽셀에 대한 CSV파일 예측들   
                ex. (1, 1, 2, 2, 3, 3 ,3)
        - 출력: 해당 예측들 중 다수결           
                ex. 3
    '''
    classes = [0] * 11
    for j, prediction in enumerate(pixel_predictions):
        classes[prediction] = classes[prediction] + weight[j]

    m = max(classes)
    max_weight_index = weight.index(max(weight))
    i = [j for j, v in enumerate(classes) if v==m]
    # 전부 베터리가 아니면 일반쓰레기
    if (9 in i):
        return 1
    # 동률일 때 weight가 높은걸로
    if len(i) > 1:
        return pixel_predictions[max_weight_index]
    # 다수결이 배경이지만 weight가 높은게 배경이 아닐경우
    elif i[0]==0 and pixel_predictions[max_weight_index] != 0:
        return pixel_predictions[max_weight_index]
    else:
        return i[0]

def main(args):
    files = args.file_list
    files = [pd.read_csv(file) for file in files]
    if args.Weight is None:
        Weights = [1] * len(files)
    if args.method=='majority':
        Weights = [1] * len(files)
    elif args.method=='weighted_majority':
        Weights = args.Weight
    elif args.method=='softmax':
        Weights = torch.nn.Softmax(dim=-1)(torch.tensor(args.Weight)).tolist()


    imgID = files[0]['image_id']
    PredictionStrings = [file['PredictionString'] for file in files]
    Stacked_Predictions = []
    print("Currently, the contents of the input files are being collected...")
    for PredictionString in tqdm(PredictionStrings):
        PredictionString = PredictionString.apply(lambda x:list(map(int, x.split())))
        Stacked_Predictions.append(PredictionString.to_numpy().tolist())
    Stacked_Predictions = np.array(Stacked_Predictions)
    Stacked_Predictions = np.transpose(Stacked_Predictions, (1, 2, 0))

    result = []
    print("ensemble...")
    for img in tqdm(Stacked_Predictions):
        temp = []
        for prediction in img:
            temp.append(str(majority(prediction, Weights)))
        temp = ' '.join(temp)
        result.append(temp)

    submission = pd.DataFrame()
    submission['image_id'] = imgID
    submission['PredictionString'] = result
    submission.to_csv(f'{args.output_file_name}.csv', index=False)

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is None:
        return v
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" %(s))
    return v

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file_list', type=arg_as_list)
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--method', type=str, help="you can choose [majority, weighted_majority, softmax...]")
    parser.add_argument('--Weight', type=arg_as_list, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)