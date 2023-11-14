import argparse
import torch
import random
import time
import os
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    total = 0
    correct_list = []        
    for i, data in enumerate(dataloader):

        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data

        x = x[0]
        y = y[0].strip()
        
        
        #x = x + "\nLet's think step by step." 
        z = decoder.decode(args, x)

        z2 = x + "\n" + z + args.direct_answer_trigger
        pred = decoder.decode(args, z2)
        print(z2 + pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        # We clean the pred and GT here!
        pred = clean_pred(pred)
        y = clean_ans(y)

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

def clean_ans(ans):
    new_ans = ""
    for i in range(len(ans)):
        if ans[i] == ",":
            continue
        new_ans += ans[i]
    # print(ans, new_ans)

    if '.' in new_ans:
        pos = new_ans.find('.')
        if len(new_ans) - pos -1 > 7:
            new_ans = new_ans[:pos + 7]
    return new_ans
    
def clean_pred(pred):
    if '.' in pred:
        pred = pred.rstrip('0')
        if pred.endswith('.'):
            pred = pred[:-1]

    if '.' in pred:
        pos = pred.find('.')
        if len(pred) - pos -1 > 7:
            pred = pred[:pos + 7]
    return pred
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # parser.add_argument(
    #     "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    # )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["turbo","gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="role_play", choices=["zero_shot", "role_play"], help="method"
    )
    # parser.add_argument(
    #     "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    # )
    # parser.add_argument(
    #     "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    # )
    # parser.add_argument(
    #     "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    # )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    # parser.add_argument(
    #     "--api_time_interval", type=float, default=1.0, help=""
    # )
    # parser.add_argument(
    #     "--log_dir", type=str, default="./log/", help="log directory"
    # )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the final answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    # args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
        
    # if args.cot_trigger_no == 1:
    #     args.cot_trigger = "Let's think step by step."
    # elif args.cot_trigger_no == 2:
    #     args.cot_trigger = "We should think about this step by step."
    # elif args.cot_trigger_no == 3:
    #     args.cot_trigger = "First,"
    # elif args.cot_trigger_no == 4:
    #     args.cot_trigger = "Before we dive into the answer,"
    # elif args.cot_trigger_no == 5:
    #     args.cot_trigger = "Proof followed by the answer."
    # elif args.cot_trigger_no == 6:
    #     args.cot_trigger = "Let's think step by step in a realistic way."
    # elif args.cot_trigger_no == 7:
    #     args.cot_trigger = "Let's think step by step using common sense and knowledge."
    # elif args.cot_trigger_no == 8:
    #     args.cot_trigger = "Let's think like a detective step by step."
    # elif args.cot_trigger_no == 9:
    #     args.cot_trigger = "Let's think about this logically."
    # elif args.cot_trigger_no == 10:
    #     args.cot_trigger = "Let's think step by step. First,"
    # elif args.cot_trigger_no == 11:
    #     args.cot_trigger = "Let's think"
    # elif args.cot_trigger_no == 12:
    #     args.cot_trigger = "Let's solve this problem by splitting it into steps."
    # elif args.cot_trigger_no == 13:
    #     args.cot_trigger = "The answer is after the proof."
    # elif args.cot_trigger_no == 14:
    #     args.cot_trigger = "Let's be realistic and think step by step."
    # else:
    #     raise ValueError("cot_trigger_no is not properly defined ...")
    
    if args.dataset in ["aqua", "svamp", "singleeq", "addsub", "gsm8k", "multiarith"]: 
        args.role_setting = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."
        args.reply = "That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you. Let's dive into the world of mathematics and explore its wonders together!"
    elif args.dataset in ["bigbench_date"]:
        args.role_setting = "From now on, you are an excellent teacher and are teaching your students how to calculate dates correctly. I am one of your students and want to ask you a related question."
        args.reply = "Of course! I'm here to help you with any questions you have about calculating dates correctly. Please go ahead and ask your question, and I'll do my best to assist you."
    elif args.dataset in ["coin_flip"]:
        args.role_setting = "From now on, you are a coin that always clearly knows which side of your head is facing. Some people want to play a game with you. They may flip you (a coin) or not. And you will tell them if you (a coin) are heads up in the end."
        args.reply = "Certainly! I'll be your coin for this game. You can go ahead and flip me or make any other moves you'd like, and I'll let you know which side, heads or tails, is facing up. Feel free to start whenever you're ready!"
    elif args.dataset in ["last_letters"]:
        args.role_setting = "From now on, you are an excellent teacher and are teaching your students to get a new word by concatenating the last letters of several words. I am one of your students and want to ask you a related question."
        args.reply = "Of course! I'd be happy to help you with any questions you have about creating new words by concatenating the last letters of several words. Please go ahead and ask your question, and I'll do my best to assist you."
    elif args.dataset in ["object_tracking"]:
        args.role_setting = "From now on, you are a recorder. Alice, Bob, and Claire invite you to record a game. They will exchange their stuff in order, and you (the recorder) will fully record the whole process and tell them what they end up with."
        args.reply = "Certainly! I will act as a recorder and document the game in which Alice, Bob, and Claire will exchange their items. Please provide me with the specific order in which they will exchange their belongings, and I will keep track of the process and inform you of what each person ends up with at the end."
    elif args.dataset in ["commonsensqa", "strategyqa"]:
        args.role_setting = "From now on, you are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. I am the moderator of the game and the final is about to start."
        args.reply = "That sounds like an exciting challenge! I'm ready to participate in the quiz contest as a contestant. Please go ahead and start the final round—I'm here to provide accurate answers to your common sense questions."
    return args

if __name__ == "__main__":
    main()