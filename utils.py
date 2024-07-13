import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, TFT5EncoderModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch.autograd import Variable
import torch.nn as nn
from conformal_prediction import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy.stats import entropy

#prompt tuning libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, Trainer
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, TaskType, PeftConfig, PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import Adafactor, AdafactorSchedule
import pandas as pd
from huggingface_hub import notebook_login

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        elif 'gptj' in params['model']:
            return 1
        elif 'llama2_13b' in params['model']:
            return 1
        elif 'llama2_7b' in params['model']:
            return 1
        elif 't5' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta',
                                       'davinci-beta']
            return 20
    else:
        return bs

# def get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels, test_labels, content_free_inputs=('N/A')):
#     """Query model with content free input, return its prediction probability for each label"""
#
#     _, all_p_y = get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, content_free_inputs, test_labels, normalize=False)
#
#     p_y = np.mean(np.array(all_p_y), axis=0)
#     p_y = p_y / np.sum(p_y)  # normalize
#     return p_y

def random_sampling(sentences, labels, num, max_length=None):
    """randomly sample subset of the training pairs"""
    if max_length is not None:
        filtered_sentences = []
        filtered_labels = []
        for index in range(len(sentences)):
            if len(sentences[index]) <= max_length:
                filtered_sentences.append(sentences[index])
                filtered_labels.append(labels[index])
        sentences = filtered_sentences
        labels = filtered_labels

    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"

    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

def expected_calibration_error(samples, true_labels, M=3):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

   # keep confidences / predicted "probabilities" as they are
    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


gpt2_model = None
gpt2_tokenizer = None

def setup_gpt2(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        # print("Setting up GPT-2 model")
        # gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        # gpt2_model.eval().cuda()
        #
        # gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # # to batch generation, we pad on the left and mask those positions out.
        # gpt2_tokenizer.padding_side = "left"
        # gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        # gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        # print("Finished")

        # seed = 20230302
        # ##start new code for prompt tuning
        # np.random.seed(seed)
        # val_train_sentences, val_train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        # val_prompts = []
        # for val_sentence in val_sentences:
        #     val_prompts.append(construct_val_prompt(params, val_train_sentences, val_train_labels, val_sentence))
        # val_chunked_prompts = list(chunks(val_prompts, chunk_size_helper(params)))
        # print("val_chunked_prompts: ", val_chunked_prompts[0:5])
        # print("val_chunked_prompts: ", len(val_chunked_prompts))

        model_name_or_path = model_name

        text_column = "text"
        label_column = "text_label"
        max_length = 64
        lr = 0.0025
        num_epochs = 50
        batch_size = 16
        num_vir_tokens = 8
        prompt_init_text = "objective, subjective"

        ##0-shot
        # best = "True or False?", acc = 80.505
        # best = "Classify if the answer is True or False?", acc = 79.783
        # best = "True, False", acc = 78.3

        dataset_name = "subj"
        dataset = load_dataset("SetFit/subj")
        print("\n\nlr (PT): ", lr)
        print("batch_size (PT): ", batch_size)
        print("max_length (PT): ", max_length)
        print("num_vir_tokens (PT): ", num_vir_tokens)
        print("num_epochs (PT): ", num_epochs)
        print("prompt_init_text (PT): ", prompt_init_text)
        # print("seed (PT): ", seed)
        print("\n")


        classes = ['objective', 'subjective']
        dataset = dataset.map(
            lambda x: {"text_label": [classes[label] for label in x["label"]]},
            batched=True,
            num_proc=1,
        )


        gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if gpt2_tokenizer.pad_token_id is None:
            gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        target_max_length = max([len(gpt2_tokenizer(class_label)["input_ids"]) for class_label in classes])

        # print("target_max_length: \n\n\n", target_max_length)

        def preprocess_function(examples):
            batch_size = len(examples[text_column])

            # print("\n\n\nexamples length: ", examples['sentence1'][0:5])
            # print("\nbatch_size: ", batch_size)
            for i in range(len(examples['text'])):
                examples['text'][i] = all_train_sentences[i]
                examples['label'][i] = all_train_labels[i]
            inputs = [f"Input: {x}\nType: " for x in examples[text_column]]
            # inputs = [f"Article: {x}\nAnswer: " for x in examples[text_column]]
            # print("\n\ninputs: ", inputs[0:10])
            # print("\n\nall_train_labels: ", all_train_labels[0:5])
            targets = [str(x) for x in examples[label_column]]
            # print("\n\ntargets: ", targets[0:10])
            model_inputs = gpt2_tokenizer(inputs)
            labels = gpt2_tokenizer(targets)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] + [gpt2_tokenizer.pad_token_id]
                # print(i, sample_input_ids, label_input_ids)
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [gpt2_tokenizer.pad_token_id] * (
                        max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                    "attention_mask"
                ][i]
                labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=len(dataset["train"]),
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        # eval_dataset = processed_datasets["validation"]
        # print("processed_datasets: ", processed_datasets)
        # print("\n\neval_dataset: ", len(eval_dataset['labels']))
        print("\ntrain_dataset: ", len(train_dataset['labels']))

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
        # eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size,
        #                              pin_memory=True)

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_vir_tokens,
            prompt_tuning_init_text=prompt_init_text,
            tokenizer_name_or_path=model_name_or_path,
        )

        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

        gpt2_model = get_peft_model(gpt2_model, peft_config)
        print(gpt2_model.print_trainable_parameters())

        optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )

        device = "cuda"
        gpt2_model = gpt2_model.to(device)

        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))

        prev_acc = 0
        for epoch in range(num_epochs):
            gpt2_model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = gpt2_model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            gpt2_model.eval()
            gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
            # eval_loss = 0
            # eval_preds = []
            # for step, batch in enumerate(tqdm(eval_dataloader)):
            #     batch = {k: v.to(device) for k, v in batch.items()}
            #     with torch.no_grad():
            #         outputs = gpt2_model(**batch)
            #     loss = outputs.loss
            #     eval_loss += loss.detach().float()
            #     eval_preds.extend(
            #         gpt2_tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
            #                                     skip_special_tokens=True)
            #     )
            #
            # eval_epoch_loss = eval_loss / len(eval_dataloader)
            # eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)

            # start computing accuracy
            all_raw_answers = []
            for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
                with torch.no_grad():
                    _, resp = complete_gpt2(test_chunk_prompts, params['label_dict'], normalize=True)

                for answer_id, answer in enumerate(resp):
                    all_raw_answers.append(answer)

            all_label_probs = np.asarray(all_raw_answers)

            num_classes = all_label_probs.shape[1]
            # content_free_inputs = ["N/A", "", "[MASK]"]
            # p_cf = get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels, test_labels,
            #                           content_free_inputs=content_free_inputs)
            p_cf = None
            if p_cf is None:
                # do not calibrate
                W = Variable(torch.eye(num_classes), requires_grad=True)
                b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
            else:
                W = Variable(torch.inverse(torch.eye(num_classes) * torch.tensor(p_cf)), requires_grad=True)
                b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)

            correctness_list = []
            assert len(all_label_probs) == len(test_labels)
            for label_probs, true_label in zip(all_label_probs, test_labels):
                label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1

                calibrate_label_probs = torch.matmul(W.float(),
                                                     torch.unsqueeze(label_probs, dim=-1).float()) + b.float()

                ans_label = torch.argmax(calibrate_label_probs)

                if ans_label == true_label:
                    correctness_list.append(1)
                else:
                    correctness_list.append(0)

            test_acc = round(np.mean(correctness_list), 5)

            checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_cep{epoch}_acc{test_acc}.pt".replace(
                "/", "_"
            )
            if (test_acc > prev_acc):
                prev_acc = test_acc
                gpt2_model.save_pretrained("saved_models/{}".format(checkpoint_name))
            # end computing accuracy

            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {test_acc=}")
        # end new code for prompt tuning

        checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_last_acc{test_acc}.pt".replace(
            "/", "_"
        )
        gpt2_model.save_pretrained("saved_models/{}".format(checkpoint_name))




gptj_model = None
gptj_tokenizer = None
def setup_gptj(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global gptj_model
    global gptj_tokenizer
    if gptj_model is None:
        print("Setting up GPT-J model")
        # folder_name = "saved_models/sample_epoch_table/agnews_EleutherAI_gpt-j-6B_0shot_trsz7000_lr0.00035_tkn8_tep35_cep34_acc0.14583.pt"
        # config = PeftConfig.from_pretrained(folder_name)
        gptj_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # gptj_model = PeftModel.from_pretrained(gptj_model, folder_name)
        gptj_model.eval().cuda()
        gptj_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # to batch generation, we pad on the left and mask those positions out.
        gptj_tokenizer.padding_side = "left"
        gptj_tokenizer.pad_token = gptj_tokenizer.eos_token
        gptj_model.config.pad_token_id = gptj_model.config.eos_token_id
        print("Finished")

        # # seed = 20230302
        # # ##start new code for prompt tuning
        # # np.random.seed(seed)
        # # val_train_sentences, val_train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        # # val_prompts = []
        # # for val_sentence in val_sentences:
        # #     val_prompts.append(construct_val_prompt(params, val_train_sentences, val_train_labels, val_sentence))
        # # val_chunked_prompts = list(chunks(val_prompts, chunk_size_helper(params)))
        # # print("val_chunked_prompts: ", val_chunked_prompts[0:5])
        # # print("val_chunked_prompts: ", len(val_chunked_prompts))

        # model_name_or_path = "EleutherAI/gpt-j-6B"

        # text_column = "text"
        # label_column = "text_label"
        # max_length = 64
        # lr = 0.00035
        # num_epochs = 15
        # batch_size = 16
        # num_vir_tokens = 8
        # prompt_init_text = "World, Sports, Business, Technology"

        # ##0-shot
        # # best = "True or False?", acc = 80.505
        # # best = "Classify if the answer is True or False?", acc = 79.783
        # # best = "True, False", acc = 78.3

        # dataset_name = "agnews"
        # dataset = load_dataset("ag_news")
        # print("\n\nlr (PT): ", lr)
        # print("batch_size (PT): ", batch_size)
        # print("max_length (PT): ", max_length)
        # print("num_vir_tokens (PT): ", num_vir_tokens)
        # print("num_epochs (PT): ", num_epochs)
        # print("prompt_init_text (PT): ", prompt_init_text)
        # # print("seed (PT): ", seed)
        # print("\n")

        # # print("\n\ndataset: ", dataset["train"]["label"][0:10])
        # # print("\n\ndataset: ", dataset["train"]["text"][0:10])

        # # classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
        # classes = ['World', 'Sports', 'Business', 'Technology']
        # # print("\n\nclasses: ", classes)
        # dataset = dataset.map(
        #     lambda x: {"text_label": [classes[label] for label in x["label"]]},
        #     batched=True,
        #     num_proc=1,
        # )

        # # print("\n\ndataset: ", len(dataset["train"]))
        # # dataset["train"] = dataset["train"].shuffle(seed=42)
        # dataset["train"] = dataset["train"].select(range(tr_dataset_size))
        # dataset["test"] = dataset["test"].select(range(tr_dataset_size))
        # # print("\n\ndataset_train: ", len(dataset["train"]))
        # # print("\n\ndataset_test: ", len(dataset["test"]))
        # # print("\n\ndatasetx: ", all_train_sentences[1350:1355])
        # # print("\n\ndatasetx: ", all_train_labels[1350:1355])

        # gptj_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # if gptj_tokenizer.pad_token_id is None:
        #     gptj_tokenizer.pad_token_id = gptj_tokenizer.eos_token_id
        # gptj_tokenizer.padding_side = "left"
        # gptj_tokenizer.pad_token = gptj_tokenizer.eos_token
        # target_max_length = max([len(gptj_tokenizer(class_label)["input_ids"]) for class_label in classes])

        # # print("target_max_length: \n\n\n", target_max_length)

        # prompts = []
        # y_test = []
        # num_shots = 0
        # for i in range(len(D_full.y)):
        #     test_str = "8APSK signals are as follows:"
        #     for j in range(num_shots):
        #         test_str += "\nSignal#{}'s real part is ".format(j + 1) + str(
        #             np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(
        #             np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        #     test_str += "\nTest Signal's real part is " + str(
        #         np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(
        #         np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
        #     prompts.append(test_str)
        #     y_test.append(D_full.y.numpy()[i])

        # def preprocess_function(examples):
        #     batch_size = len(examples[text_column])

        #     # print("\n\n\nexamples length: ", examples['sentence1'][0:5])
        #     # print("\n\n\nbatch_size: ", batch_size)
        #     for i in range(len(examples[text_column])):
        #         examples['text'][i] = prompts[i]
        #         examples['label'][i] = y_test[i]
        #     inputs = [f"{x}" for x in examples[text_column]]
        #     # inputs = [f"Article: {x}\nAnswer: " for x in examples[text_column]]
        #     # print("\n\ninputs: ", inputs[0:5])
        #     targets = [str(x) for x in y_test]
        #     # print("\n\ntargets: ", targets[0:5])
        #     model_inputs = gptj_tokenizer(inputs)
        #     labels = gptj_tokenizer(targets)
        #     for i in range(batch_size):
        #         sample_input_ids = model_inputs["input_ids"][i]
        #         label_input_ids = labels["input_ids"][i] + [gptj_tokenizer.pad_token_id]
        #         # print(i, sample_input_ids, label_input_ids)
        #         model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        #         labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        #         model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        #     for i in range(batch_size):
        #         sample_input_ids = model_inputs["input_ids"][i]
        #         label_input_ids = labels["input_ids"][i]
        #         model_inputs["input_ids"][i] = [gptj_tokenizer.pad_token_id] * (
        #                 max_length - len(sample_input_ids)
        #         ) + sample_input_ids
        #         model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
        #             "attention_mask"
        #         ][i]
        #         labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        #         model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        #         model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        #         labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        #     model_inputs["labels"] = labels["input_ids"]
        #     return model_inputs

        # processed_datasets = dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     batch_size=len(dataset["train"]),
        #     num_proc=1,
        #     remove_columns=dataset["train"].column_names,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on dataset",
        # )

        # train_dataset = processed_datasets["train"]
        # # eval_dataset = processed_datasets["validation"]
        # # print("processed_datasets: ", processed_datasets)
        # # print("\n\neval_dataset: ", len(eval_dataset['labels']))
        # print("\ntrain_dataset: ", len(train_dataset['labels']))

        # train_dataloader = DataLoader(
        #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        # )
        # # eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size,
        # #                              pin_memory=True)

        # peft_config = PromptTuningConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     prompt_tuning_init=PromptTuningInit.TEXT,
        #     num_virtual_tokens=num_vir_tokens,
        #     prompt_tuning_init_text=prompt_init_text,
        #     tokenizer_name_or_path=model_name_or_path,
        # )

        # gptj_model = GPTJForCausalLM.from_pretrained(model_name_or_path, revision="float16",
        #                                              torch_dtype=torch.float16, low_cpu_mem_usage=True)

        # gptj_model = get_peft_model(gptj_model, peft_config)
        # print(gptj_model.print_trainable_parameters())

        # optimizer = torch.optim.AdamW(gptj_model.parameters(), lr=lr)
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=(len(train_dataloader) * num_epochs),
        # )

        # device = "cuda"
        # gptj_model = gptj_model.to(device)

        # prompts_test = []
        # y_test_acc = []
        # num_shots = 0
        # for i in range(len(D_te.y)):
        #     test_str = "8APSK signals are as follows:"
        #     for j in range(num_shots):
        #         test_str += "\nSignal#{}'s real part is ".format(j + 1) + str(
        #             np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(
        #             np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        #     test_str += "\nTest Signal's real part is " + str(
        #         np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(
        #         np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
        #     prompts_test.append(test_str)
        #     y_test_acc.append(D_te.y.numpy()[i])

        # chunked_prompts = list(chunks(prompts_test, chunk_size_helper(params)))

        # prev_acc = 0
        # for epoch in range(num_epochs):
        #     gptj_model.train()
        #     total_loss = 0
        #     for step, batch in enumerate(tqdm(train_dataloader)):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         outputs = gptj_model(**batch)
        #         loss = outputs.loss
        #         total_loss += loss.detach().float()
        #         loss.backward()
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()

        #     gptj_model.eval()
        #     gptj_model.config.pad_token_id = gptj_model.config.eos_token_id
        #     # eval_loss = 0
        #     # eval_preds = []
        #     # for step, batch in enumerate(tqdm(eval_dataloader)):
        #     #     batch = {k: v.to(device) for k, v in batch.items()}
        #     #     with torch.no_grad():
        #     #         outputs = gptj_model(**batch)
        #     #     loss = outputs.loss
        #     #     eval_loss += loss.detach().float()
        #     #     eval_preds.extend(
        #     #         gptj_tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
        #     #                                     skip_special_tokens=True)
        #     #     )
        #     #
        #     # eval_epoch_loss = eval_loss / len(eval_dataloader)
        #     # eval_ppl = torch.exp(eval_epoch_loss)
        #     train_epoch_loss = total_loss / len(train_dataloader)
        #     train_ppl = torch.exp(train_epoch_loss)

        #     # start computing accuracy
        #     all_raw_answers = []
        #     for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        #         with torch.no_grad():
        #             _, resp = complete_gptj(test_chunk_prompts, params['label_dict'], normalize=True)

        #         for answer_id, answer in enumerate(resp):
        #             all_raw_answers.append(answer)

        #     all_label_probs = np.asarray(all_raw_answers)

        #     num_classes = all_label_probs.shape[1]
        #     # content_free_inputs = ["N/A", "", "[MASK]"]
        #     # p_cf = get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels, test_labels,
        #     #                           content_free_inputs=content_free_inputs)
        #     p_cf = None
        #     if p_cf is None:
        #         # do not calibrate
        #         W = Variable(torch.eye(num_classes), requires_grad=True)
        #         b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
        #     else:
        #         W = Variable(torch.inverse(torch.eye(num_classes) * torch.tensor(p_cf)), requires_grad=True)
        #         b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)

        #     correctness_list = []
        #     assert len(all_label_probs) == len(y_test_acc)
        #     for label_probs, true_label in zip(all_label_probs, y_test_acc):
        #         label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1

        #         calibrate_label_probs = torch.matmul(W.float(),
        #                                              torch.unsqueeze(label_probs, dim=-1).float()) + b.float()

        #         ans_label = torch.argmax(calibrate_label_probs)

        #         if ans_label == true_label:
        #             correctness_list.append(1)
        #         else:
        #             correctness_list.append(0)

        #     test_acc = round(np.mean(correctness_list), 5)

        #     dataset_name = 'demodulation'
        #     checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_cep{epoch}_acc{test_acc}.pt".replace(
        #         "/", "_"
        #     )
        #     # if (test_acc > prev_acc):
        #     # prev_acc = test_acc
        #     gptj_model.save_pretrained("saved_models/sample_epoch_table/{}".format(checkpoint_name))
        #     # end computing accuracy

        #     print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {test_acc=}")
        # # end new code for prompt tuning

        # checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_last_acc{test_acc}.pt".replace(
        #     "/", "_"
        # )
        # gptj_model.save_pretrained("saved_models/{}".format(checkpoint_name))


llamma2_7b_model = None
llamma2_7b_tokenizer  = None
def setup_llama2_7b(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global llamma2_7b_model
    global llamma2_7b_tokenizer
    if llamma2_7b_model is None:
        print("Setting up Llama-2 7b model")
        # folder_name = "saved_models/sample_epoch_table/agnews/agnews_meta-llama_Llama-2-7b-hf_0shot_trsz6000_lr0.00035_tkn8_tep35_cep14_acc0.87333.pt"
        model_name = 'meta-llama/Llama-2-7b-hf'
        
        config = AutoConfig.from_pretrained(model_name)
        config.pretraining_tp = 1
        llamma2_7b_model = LlamaForCausalLM.from_pretrained(model_name, device_map='sequential', config=config,
                                                             torch_dtype=torch.float16,
                                                             low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        
        # llamma2_7b_model = PeftModel.from_pretrained(llamma2_7b_model, folder_name)
        
        llamma2_7b_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        llamma2_7b_tokenizer.padding_side = "left"
        llamma2_7b_tokenizer.pad_token = llamma2_7b_tokenizer.eos_token
        llamma2_7b_model.config.pad_token_id = llamma2_7b_model.config.eos_token_id
        print("Finished")


llamma2_13b_model = None
llamma2_13b_tokenizer  = None
def setup_llama2_13b(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global llamma2_13b_model
    global llamma2_13b_tokenizer
    if llamma2_13b_model is None:
        print("Setting up Llama-2 13B model")
        # folder_name = "saved_models/sample_epoch_table/agnews/agnews_meta-llama_Llama-2-13b-hf_0shot_trsz6000_lr0.00035_tkn8_tep35_cep14_acc0.87333.pt"
        model_name = 'meta-llama/Llama-2-13b-hf'
        
        device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
                      'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
                      'model.layers.7': 0,
                      'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
                      'model.layers.12': 0,
                      'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0,
                      'model.layers.17': 0,
                      'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
                      'model.layers.22': 1,
                      'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1,
                      'model.layers.27': 1,
                      'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
                      'model.layers.32': 1,
                      'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1,
                      'model.layers.37': 1,
                      'model.layers.38': 2, 'model.layers.39': 2, 'model.norm': 2, 'lm_head': 2}
        
        config = AutoConfig.from_pretrained(model_name)
        config.pretraining_tp = 1
        llamma2_13b_model = LlamaForCausalLM.from_pretrained(model_name, device_map=device_map, config=config,
                                                             torch_dtype=torch.float16,
                                                             low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        
        # llamma2_13b_model = PeftModel.from_pretrained(llamma2_13b_model, folder_name)
        
        llamma2_13b_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        llamma2_13b_tokenizer.padding_side = "left"
        llamma2_13b_tokenizer.pad_token = llamma2_13b_tokenizer.eos_token
        llamma2_13b_model.config.pad_token_id = llamma2_13b_model.config.eos_token_id
        print("Finished")

        # # seed = 20230302
        # # ##start new code for prompt tuning
        # # np.random.seed(seed)
        # # val_train_sentences, val_train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        # # val_prompts = []
        # # for val_sentence in val_sentences:
        # #     val_prompts.append(construct_val_prompt(params, val_train_sentences, val_train_labels, val_sentence))
        # # val_chunked_prompts = list(chunks(val_prompts, chunk_size_helper(params)))
        # # print("val_chunked_prompts: ", val_chunked_prompts[0:5])
        # # print("val_chunked_prompts: ", len(val_chunked_prompts))

        # model_name = 'meta-llama/Llama-2-7b-hf'
        # model_name_or_path = model_name

        # text_column = "text"
        # label_column = "text_label"
        # max_length = 64
        # lr = 0.00035
        # num_epochs = 26
        # batch_size = 16
        # num_vir_tokens = 8
        # prompt_init_text = "World, Sports, Business, Technology"

        # ##0-shot
        # # best = "True or False?", acc = 80.505
        # # best = "Classify if the answer is True or False?", acc = 79.783
        # # best = "True, False", acc = 78.3

        # dataset_name = "agnews"
        # dataset = load_dataset("ag_news")
        # print("\n\nlr (PT): ", lr)
        # print("batch_size (PT): ", batch_size)
        # print("max_length (PT): ", max_length)
        # print("num_vir_tokens (PT): ", num_vir_tokens)
        # print("num_epochs (PT): ", num_epochs)
        # print("prompt_init_text (PT): ", prompt_init_text)
        # # print("seed (PT): ", seed)
        # print("\n")

        # # print("\n\ndataset: ", dataset["train"]["label"][0:10])
        # # print("\n\ndataset: ", dataset["train"]["text"][0:10])

        # # classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
        # classes = ['World', 'Sports', 'Business', 'Technology']
        # # print("\n\nclasses: ", classes)
        # dataset = dataset.map(
        #     lambda x: {"text_label": [classes[label] for label in x["label"]]},
        #     batched=True,
        #     num_proc=1,
        # )

        # # print("\n\ndataset: ", len(dataset["train"]))
        # # dataset["train"] = dataset["train"].shuffle(seed=42)
        # dataset["train"] = dataset["train"].select(range(tr_dataset_size))
        # dataset["test"] = dataset["test"].select(range(tr_dataset_size))
        # # print("\n\ndatasetw: ", dataset["train"][0:5])
        # # print("\n\ndatasetx: ", all_train_sentences[1350:1355])
        # # print("\n\ndatasetx: ", all_train_labels[1350:1355])

        # llamma2_13b_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # # to batch generation, we pad on the left and mask those positions out.
        # llamma2_13b_tokenizer.padding_side = "left"
        # llamma2_13b_tokenizer.pad_token = llamma2_13b_tokenizer.eos_token
        # target_max_length = max([len(llamma2_13b_tokenizer(class_label)["input_ids"]) for class_label in classes])

        # prompts = []
        # y_test = []
        # num_shots = 0
        # for i in range(len(D_full.y)):
        #     test_str = "8APSK signals are as follows:"
        #     for j in range(num_shots):
        #         test_str += "\nSignal#{}'s real part is ".format(j + 1) + str(
        #             np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(
        #             np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        #     test_str += "\nTest Signal's real part is " + str(
        #         np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(
        #         np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
        #     prompts.append(test_str)
        #     y_test.append(D_full.y.numpy()[i])

        # def preprocess_function(examples):
        #     batch_size = len(examples[text_column])

        #     # print("\n\n\nexamples length: ", examples['sentence1'][0:5])
        #     # print("\n\n\nbatch_size: ", batch_size)
        #     for i in range(len(examples[text_column])):
        #         examples['text'][i] = prompts[i]
        #         examples['label'][i] = y_test[i]
        #     inputs = [f"{x}" for x in examples[text_column]]
        #     # inputs = [f"Article: {x}\nAnswer: " for x in examples[text_column]]
        #     # print("\n\ninputs: ", inputs[0:5])
        #     targets = [str(x) for x in y_test]
        #     # print("\n\ntargets: ", targets[0:5])
        #     model_inputs = llamma2_13b_tokenizer(inputs)
        #     labels = llamma2_13b_tokenizer(targets)
        #     for i in range(batch_size):
        #         sample_input_ids = model_inputs["input_ids"][i]
        #         label_input_ids = labels["input_ids"][i] + [llamma2_13b_tokenizer.pad_token_id]
        #         # print(i, sample_input_ids, label_input_ids)
        #         model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        #         labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        #         model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        #     for i in range(batch_size):
        #         sample_input_ids = model_inputs["input_ids"][i]
        #         label_input_ids = labels["input_ids"][i]
        #         model_inputs["input_ids"][i] = [llamma2_13b_tokenizer.pad_token_id] * (
        #                 max_length - len(sample_input_ids)
        #         ) + sample_input_ids
        #         model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
        #             "attention_mask"
        #         ][i]
        #         labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        #         model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        #         model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        #         labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        #     model_inputs["labels"] = labels["input_ids"]
        #     return model_inputs

        # processed_datasets = dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     batch_size=len(dataset["train"]),
        #     num_proc=1,
        #     remove_columns=dataset["train"].column_names,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on dataset",
        # )

        # train_dataset = processed_datasets["train"]
        # # eval_dataset = processed_datasets["validation"]
        # # print("processed_datasets: ", processed_datasets)
        # # print("\n\neval_dataset: ", len(eval_dataset['labels']))
        # print("\ntrain_dataset: ", len(train_dataset['labels']))

        # train_dataloader = DataLoader(
        #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        # )
        # # eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size,
        # #                              pin_memory=True)

        # peft_config = PromptTuningConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     prompt_tuning_init=PromptTuningInit.TEXT,
        #     num_virtual_tokens=num_vir_tokens,
        #     prompt_tuning_init_text=prompt_init_text,
        #     tokenizer_name_or_path=model_name,
        # )

        # # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
        # #               'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
        # #               'model.layers.7': 0,
        # #               'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
        # #               'model.layers.12': 0,
        # #               'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0,
        # #               'model.layers.17': 0,
        # #               'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
        # #               'model.layers.22': 1,
        # #               'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1,
        # #               'model.layers.27': 1,
        # #               'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
        # #               'model.layers.32': 1,
        # #               'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1,
        # #               'model.layers.37': 1,
        # #               'model.layers.38': 1, 'model.layers.39': 1, 'model.norm': 1, 'lm_head': 1}

        # # config = AutoConfig.from_pretrained(model_name)
        # # config.pretraining_tp = 1
        # # llamma2_13b_model = LlamaForCausalLM.from_pretrained(model_name, device_map=device_map, config=config,
        # #                                                      torch_dtype=torch.float16,
        # #                                                      low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        # config = AutoConfig.from_pretrained(model_name)
        # config.pretraining_tp = 1
        # llamma2_13b_model = LlamaForCausalLM.from_pretrained(model_name, config=config,
        #                                                      torch_dtype=torch.float16,
        #                                                      low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling

        # llamma2_13b_model.config.pad_token_id = llamma2_13b_model.config.eos_token_id
        # llamma2_13b_model = get_peft_model(llamma2_13b_model, peft_config)
        # print(llamma2_13b_model.print_trainable_parameters())

        # optimizer = torch.optim.AdamW(llamma2_13b_model.parameters(), lr=lr)
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=(len(train_dataloader) * num_epochs),
        # )

        # device = "cuda"
        # # llamma2_13b_model = llamma2_13b_model.to(device)

        # prompts_test = []
        # y_test_acc = []
        # num_shots = 0
        # for i in range(len(D_te.y)):
        #     test_str = "8APSK signals are as follows:"
        #     for j in range(num_shots):
        #         test_str += "\nSignal#{}'s real part is ".format(j + 1) + str(
        #             np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(
        #             np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        #     test_str += "\nTest Signal's real part is " + str(
        #         np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(
        #         np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
        #     prompts_test.append(test_str)
        #     y_test_acc.append(D_te.y.numpy()[i])

        # chunked_prompts = list(chunks(prompts_test, chunk_size_helper(params)))

        # print('\n\n\ntrain_dataloader: ', train_dataloader)

        # prev_acc = 0
        # for epoch in range(num_epochs):
        #     llamma2_13b_model.train()
        #     total_loss = 0
        #     for step, batch in enumerate(tqdm(train_dataloader)):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         outputs = llamma2_13b_model(**batch)
        #         loss = outputs.loss
        #         total_loss += loss.detach().float()
        #         loss.backward()
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()

        #     llamma2_13b_model.eval()
        #     llamma2_13b_model.config.pad_token_id = llamma2_13b_model.config.eos_token_id
        #     # eval_loss = 0
        #     # eval_preds = []
        #     # for step, batch in enumerate(tqdm(eval_dataloader)):
        #     #     batch = {k: v.to(device) for k, v in batch.items()}
        #     #     with torch.no_grad():
        #     #         outputs = llamma2_13b_model(**batch)
        #     #     loss = outputs.loss
        #     #     eval_loss += loss.detach().float()
        #     #     eval_preds.extend(
        #     #         gptj_tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
        #     #                                     skip_special_tokens=True)
        #     #     )
        #     #
        #     # eval_epoch_loss = eval_loss / len(eval_dataloader)
        #     # eval_ppl = torch.exp(eval_epoch_loss)
        #     train_epoch_loss = total_loss / len(train_dataloader)
        #     train_ppl = torch.exp(train_epoch_loss)

        #     # start computing accuracy
        #     all_raw_answers = []
        #     for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        #         with torch.no_grad():
        #             _, resp = complete_gptj(test_chunk_prompts, params['label_dict'], normalize=True)

        #         for answer_id, answer in enumerate(resp):
        #             all_raw_answers.append(answer)

        #     all_label_probs = np.asarray(all_raw_answers)

        #     num_classes = all_label_probs.shape[1]
        #     # content_free_inputs = ["N/A", "", "[MASK]"]
        #     # p_cf = get_p_content_free(params, all_train_sentences, all_train_labels, train_sentences, train_labels, test_labels,
        #     #                           content_free_inputs=content_free_inputs)
        #     p_cf = None
        #     if p_cf is None:
        #         # do not calibrate
        #         W = Variable(torch.eye(num_classes), requires_grad=True)
        #         b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
        #     else:
        #         W = Variable(torch.inverse(torch.eye(num_classes) * torch.tensor(p_cf)), requires_grad=True)
        #         b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)

        #     correctness_list = []
        #     assert len(all_label_probs) == len(y_test_acc)
        #     for label_probs, true_label in zip(all_label_probs, y_test_acc):
        #         label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1

        #         calibrate_label_probs = torch.matmul(W.float(),
        #                                              torch.unsqueeze(label_probs, dim=-1).float()) + b.float()

        #         ans_label = torch.argmax(calibrate_label_probs)

        #         if ans_label == true_label:
        #             correctness_list.append(1)
        #         else:
        #             correctness_list.append(0)

        #     test_acc = round(np.mean(correctness_list), 5)

        #     dataset_name = 'demodulation'
        #     checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_cep{epoch}_acc{test_acc}.pt".replace(
        #         "/", "_"
        #     )
        #     # if (test_acc > prev_acc):
        #     # prev_acc = test_acc
        #     llamma2_13b_model.save_pretrained("saved_models/sample_epoch_table/{}".format(checkpoint_name))
        #     # end computing accuracy

        #     print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {test_acc=}")
        # # end new code for prompt tuning

        # checkpoint_name = f"{dataset_name}_{model_name_or_path}_{params['num_shots']}shot_trsz{len(train_dataset['labels'])}_lr{lr}_tkn{num_vir_tokens}_tep{num_epochs}_last_acc{test_acc}.pt".replace(
        #     "/", "_"
        # )
        # llamma2_13b_model.save_pretrained("saved_models/{}".format(checkpoint_name))

# def setup_gptj(model_name):
#     # load the GPT-2 model
#     global gptj_model
#     global gptj_tokenizer
#     if gptj_model is None:
#         print("Setting up GPT-J model")
#         # gptj_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
#
#         checkpoint = "EleutherAI/gpt-j-6B"
#         config = AutoConfig.from_pretrained(checkpoint)
#         with init_empty_weights():
#             gptj_model = AutoModelForCausalLM.from_config(config)
#
#         gptj_model.tie_weights()
#
#         gptj_model = load_checkpoint_and_dispatch(
#             gptj_model, "sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"]
#         )
#
#         # gptj_model.eval().cuda()
#         gptj_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#         # to batch generation, we pad on the left and mask those positions out.
#         gptj_tokenizer.padding_side = "left"
#         gptj_tokenizer.pad_token = gptj_tokenizer.eos_token
#         gptj_model.config.pad_token_id = gptj_model.config.eos_token_id
#
#         print("Finished")


# t5_model = None
# t5_tokenizer = None
# def setup_t5(model_name, params, train_sentences, train_labels, test_sentences, test_labels):
#     # load the t5 model
#     global t5_model
#     global t5_tokenizer
#     if t5_model is None:
#         print("Setting up T5 model")
#         t5_model = TFT5EncoderModel.from_pretrained("t5-small")
#         t5_model.eval().cuda()
#
#         t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
#         # to batch generation, we pad on the left and mask those positions out.
#         t5_tokenizer.padding_side = "left"
#         t5_tokenizer.pad_token = t5_tokenizer.eos_token
#         t5_model.config.pad_token_id = t5_model.config.eos_token_id
#         print("Finished")


bloomz_model = None
bloomz_tokenizer = None
def setup_bloomz(model_name):
    # load the GPT-2 model
    global bloomz_model
    global bloomz_tokenizer
    if bloomz_model is None:
        print("Setting up Bloomz model")
        # gptj_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        checkpoint = "bigscience/bloomz"
        config = AutoConfig.from_pretrained(checkpoint)
        with init_empty_weights():
            bloomz_model = AutoModelForCausalLM.from_config(config)

        bloomz_model.tie_weights()

        bloomz_model = load_checkpoint_and_dispatch(bloomz_model, "bloomz", device_map="auto", load_in_8bit=True)

        # bloomz_model.eval().cuda()
        bloomz_tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
        # to batch generation, we pad on the left and mask those positions out.
        bloomz_tokenizer.padding_side = "left"
        bloomz_tokenizer.pad_token = bloomz_tokenizer.eos_token
        bloomz_model.config.pad_token_id = bloomz_model.config.eos_token_id
        print("Finished")


def complete_gptj(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = gptj_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    if(len(input_ids['input_ids']) > 1023):
        input_ids['input_ids'] = input_ids['input_ids'][0:1023]
        input_ids['attention_mask'] = input_ids['attention_mask'][0:1023]
    total_sequences = gptj_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)

    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = gptj_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()
    # bs x 50257
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = gptj_tokenizer.encode(label)[0]
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs



def complete_llamma2_7b(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = llamma2_7b_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    total_sequences = llamma2_7b_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 31999).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = llamma2_7b_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1].float(), dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()

    # bs x 31999
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_7b_tokenizer.encode(label)[2]
            # print("token", token)
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs



def complete_llamma2_13b(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = llamma2_13b_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    total_sequences = llamma2_13b_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 31999).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = llamma2_13b_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1].float(), dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()

    # bs x 31999
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_13b_tokenizer.encode(label)[2]
            # print("token", token)
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs

def complete_bloomz(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = bloomz_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    total_sequences = bloomz_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)

    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = bloomz_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()
    # bs x 50257
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = bloomz_tokenizer.encode(label)[0]
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs


def complete_gpt2(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    if (len(input_ids['input_ids'][0]) > 1024):
        input_ids['input_ids'] = input_ids['input_ids'][:, :1023]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, :1023]
    total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)

    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens

    if (total_sequences.size(1) > 1024):
        total_sequences = total_sequences[:, :1023]
        attention_mask = attention_mask[:, :1023]
        position_ids = position_ids[:, :1023]
    logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()
    # bs x 50257
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = gpt2_tokenizer.encode(label)[0]
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs


def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function.
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l,
                                                                       np.int64):  # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str)  # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1
    return prompt

def construct_val_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function.
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l,
                                                                       np.int64):  # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str)  # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence
    # assert a_prefix[-1] == ' '
    # prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1
    return prompt

dSetting = {'snr_dB': 5.0,  # Signal to noise ratio per one RX antenna
            'modKey': '8APSK'}  # modulation key
tPercentiles = (90, 50)
iotUplink = IotUplink(dSetting)
iotUplink.draw_channel_state()  # new channel state, groundtruth c
tr_dataset_size = 24
te_dataset_size = 101
D_full = iotUplink.step(tr_dataset_size, False, False)
D_te = iotUplink.step(te_dataset_size, False, False)
count = 0

def get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels, normalize=True, key=None):
    all_raw_answers = []
    all_logits = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    # prompts = []
    # for test_sentence in test_sentences:
    #     prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))

    ###start communications code
    # torch.save(D_full.X, 'saved_data/D_full_X_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    # torch.save(D_full.y, 'saved_data/D_full_y_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    # torch.save(D_te.X, 'saved_data/D_te_X_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    # torch.save(D_te.y, 'saved_data/D_te_y_{}shots_{}.pt'.format(params['num_shots'], params['model']))

    D_full.X = torch.load('saved_data/D_full_X_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    D_full.y = torch.load('saved_data/D_full_y_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    D_te.X = torch.load('saved_data/D_te_X_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    D_te.y = torch.load('saved_data/D_te_y_{}shots_{}.pt'.format(params['num_shots'], params['model']))

    prompts = []
    y_test = []
    num_shots = params['num_shots']
    for i in range(len(D_te.y)):
        test_str  = "8APSK signals are as follows:"
        for j in range(num_shots):
            test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        test_str += "\nTest Signal's real part is " + str(np.round(D_te.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_te.X.numpy()[i][1], 3)) + ". Actual Signal: "
        prompts.append(test_str)
        y_test.append(D_te.y.numpy()[i])
        # print("\n\ntest_str: ", test_str)
    params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

    ###end communications code

    print("\n\nprompt: {}\n".format(prompts[0:1]))
    # print("\nlabel_dict: ", params['label_dict'])
    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        with torch.no_grad():
            if 'gpt2' in params['model']:
                setup_gpt2(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gpt2(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'gptj' in params['model']:
                setup_gptj(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gptj(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_13b' in params['model']:
                setup_llama2_13b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_llamma2_13b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_7b' in params['model']:
                setup_llama2_7b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_llamma2_7b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'bloomz' in params['model']:
                setup_bloomz(params['model'])
                logits, resp = complete_bloomz(test_chunk_prompts, params['label_dict'], normalize=normalize)
            else:
                raise NotImplementedError
        for answer_id, answer in enumerate(resp):
            all_raw_answers.append(answer)
        for logit in logits:
            all_logits.append(logit)

    return np.asarray(all_logits), np.asarray(all_raw_answers), y_test

def get_model_response_val(params, all_train_sentences, all_train_labels, best_train_sentences, best_train_labels, val_sentences, val_labels, test_sentences, test_labels, normalize=True, key=None):
    all_raw_answers = []
    all_logits = []
    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    # prompts = []
    # for i in range(len(test_sentences)):
    #     prompts.append(construct_prompt(params, best_train_sentences[i], best_train_labels[i], test_sentences[i]))

    ###start communications code
    D_full.X = torch.load('saved_data/D_full_X_{}shots_{}.pt'.format(params['num_shots'], params['model']))
    D_full.y = torch.load('saved_data/D_full_y_{}shots_{}.pt'.format(params['num_shots'], params['model']))

    prompts = []
    y_test = []
    num_shots = params['num_shots']
    for i in range(len(D_full.y)):
        test_str  = "8APSK signals are as follows:"
        for j in range(num_shots):
            test_str += "\nSignal#{}'s real part is ".format(j+1) + str(np.round(D_full.X.numpy()[j][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[j][1], 3)) + ". Actual Signal: {}".format(str(D_full.y.numpy()[j]))

        test_str += "\nTest Signal's real part is " + str(np.round(D_full.X.numpy()[i][0], 3)) + " and imaginery part is " + str(np.round(D_full.X.numpy()[i][1], 3)) + ". Actual Signal: "
        prompts.append(test_str)
        y_test.append(D_full.y.numpy()[i])
    params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7']}

    ###end communications code
    # print("\nval_prompt: ", prompts[0:2])
    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        with torch.no_grad():
            if 'gpt2' in params['model']:
                setup_gpt2(params['model'], params, all_train_sentences, all_train_labels, best_train_sentences, best_train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gpt2(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'gptj' in params['model']:
                setup_gptj(params['model'], params, all_train_sentences, all_train_labels, best_train_sentences, best_train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gptj(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_13b' in params['model']:
                setup_llama2_13b(params['model'], params, all_train_sentences, all_train_labels, best_train_sentences, best_train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_llamma2_13b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_7b' in params['model']:
                setup_llama2_7b(params['model'], params, all_train_sentences, all_train_labels, best_train_sentences, best_train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_llamma2_7b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'bloomz' in params['model']:
                setup_bloomz(params['model'])
                logits, resp = complete_bloomz(test_chunk_prompts, params['label_dict'], normalize=normalize)
            else:
                raise NotImplementedError

        for answer_id, answer in enumerate(resp):
            all_raw_answers.append(answer)
        for logit in logits:
            all_logits.append(logit)

    return np.asarray(all_logits), np.asarray(all_raw_answers), y_test

def params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    if 'gpt2' in params['model']:
        setup_gpt2(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
    elif 'gptj' in params['model']:
        setup_gptj(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels)
    elif 'llama2_13b' in params['model']:
        setup_llama2_13b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels)
    elif 't5' in params['model']:
        setup_t5(params['model'], params, train_sentences, train_labels, test_sentences, test_labels)
    elif 'bloomz' in params['model']:
        setup_bloomz(params['model'])
    else:
        return
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            with torch.no_grad():
                if gpt2_tokenizer is not None:
                    input_ids = gpt2_tokenizer.encode(' ' + label_name)
                    assert len(input_ids) == 1, 'label name is more than 1 token'
                elif gptj_tokenizer is not None:
                    input_ids = gptj_tokenizer.encode(' ' + label_name)
                    assert len(input_ids) == 1, 'label name is more than 1 token'
                    # print("input_ids", input_ids)
                elif llamma2_13b_tokenizer is not None:
                    input_ids = llamma2_13b_tokenizer.encode(' ' + label_name)[2]
                    assert len([input_ids]) == 1, 'label name is more than 1 token'
                elif bloomz_tokenizer is not None:
                    input_ids = bloomz_tokenizer.encode(' ' + label_name)
                    assert len(input_ids) == 1, 'label name is more than 1 token'
                else:
                    assert len(input_ids) == 1, 'label name is more than 1 token'

    if not (params['dataset'] in ['rte', 'cb']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'


def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy ','ConC Accuracy    ', 'LinC Accuracy     ')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()


def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)