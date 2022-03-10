import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from discriminator import Discriminator

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples



#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.discriminator_lr = args.adv_lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.adv_every = args.adv_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        self.adv = args.adv
        self.discriminator = Discriminator(input_dim=768)
        self.discriminator_lambda = args.adv_lambda
        self.length_loss = args.length_loss
        self.length_k = args.length_k
        self.length_lambda = args.length_lambda
        self.length_bp_penalty = args.length_bp_penalty
        self.length_mask = torch.ones(384, 384).to(self.device)
        # added
        self.weight_decay = args.weight_decay
        

        for i in range(384):
            self.length_mask[i][i:i+self.length_k] = 0

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)
        torch.save(self.discriminator.state_dict(), self.path + '/discriminator')


    # TODO: add discrinminator precision
    def evaluate(self, model, discriminator, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        # all_dis_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                hidden_states = outputs['hidden_states'][0]
                # dis_logits = torch.softmax(discriminator(hidden_states), dim=-1)
                # TODO: compute loss (do really need the loss here?)

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                # all_dis_logits.append(dis_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        # dis_logits = torch.cat(all_dis_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                            data_loader.dataset.encodings,
                            (start_logits, end_logits))
        if split == 'validation':
            # TODO : implement eval_discriminator(...)
             # discriminator_eval_results = util.eval_discriminator(data_dict, ground_truth_data_set_ids, dis_logits)
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def compute_discriminator_loss(self, hidden_states, y_domain):
        """
        Computes the loss for discriminator based on the hidden states of the DistillBERT model.
        Original paper implementation: https://github.com/seanie12/mrqa/blob/master/model.py
        https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForQuestionAnswering
        Input: last layer hidden states of the distillBERT model, with shape [batch_size, sequence_length, hidden_dim]
        :return: loss from discriminator.
        """
        embedding = hidden_states[:, 0]
        logits = self.discriminator(embedding)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_domain)
        # print('dis loss:', loss.item())
        return loss

    def train(self, model, train_dataloader, eval_dataloader, val_dict, clf_dataloader=None):
        device = self.device
        model.to(device)
        self.discriminator.to(device)
        qa_optim = AdamW(model.parameters(), lr=self.lr)
        dis_optim = AdamW(self.discriminator.parameters(), lr=self.discriminator_lr)

        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            # print(len(train_dataloader.dataset), len(eval_dataloader.dataset))
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                
                # train qa system
                print("training qa...")
                for batch in train_dataloader:
                    qa_optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, output_hidden_states=True)
                    loss = outputs[0]
                    # shape: torch.Size([16, 384, 768])
                    # print('shape:',outputs[3][0].shape)
                    start_logits, end_logits = outputs[1], outputs[2]
                    
                    if self.length_bp_penalty:
                        mean_logits = 1./2 * (start_logits + end_logits)
                        mean_positions = (1./2 * (start_positions + end_positions)).to(torch.int64)
                        penalty_loss = self.length_lambda * nn.CrossEntropyLoss()(mean_logits, mean_positions)
                        loss += penalty_loss
                    
                    if self.length_loss:
                        start_logits_softmax = F.softmax(start_logits, dim=1)
                        end_logits_softmax = F.softmax(end_logits, dim=1)
                        start_logits_softmax = torch.unsqueeze(start_logits_softmax, 2) # (batch, query_len, 1)
                        end_logits_softmax = torch.unsqueeze(end_logits_softmax, 1) # (batch, 1, query_len)
                        length_loss = torch.sum(torch.matmul(torch.bmm(start_logits_softmax, end_logits_softmax), self.length_mask)) / self.batch_size
                        loss += self.length_lambda * length_loss
                        # print('length_loss:', self.length_lambda * length_loss)
                    
                    # step the qa_optim first
                    loss.backward()
                    qa_optim.step()
                    progress_bar.update(len(input_ids))

                    # train the discriminator every eval step!
                    if self.adv and (global_idx % self.adv_every) == 0:
                        print("switching training discriminator...")
                        # hidden_states shape: [16, 384, 768]
                        # [batch_size, sequence_length, hidden_size]
                        for clf_batch, label_batch in clf_dataloader:
                            dis_optim.zero_grad()
                            model.eval()
                            self.discriminator.train()

                            label_batch = label_batch.to(device)
                            input_ids = clf_batch['input_ids'].to(device)
                            attention_mask = clf_batch['attention_mask'].to(device)
                            outputs = model(input_ids, attention_mask=attention_mask,
                                            output_hidden_states=True)

                            hidden_states = outputs['hidden_states'][0]

                            discriminator_loss = self.discriminator_lambda * self.compute_discriminator_loss(hidden_states, label_batch)
                            # print('dis loss on qa : ', discriminator_loss)
                            # step the dis_optim
                            discriminator_loss.backward()
                            dis_optim.step()

                    if self.adv:
                        tbx.add_scalar('train/dis_loss', discriminator_loss.item(), global_idx)
                        progress_bar.set_postfix(epoch=epoch_num, Loss=loss.item(), 
                        dis_loss=discriminator_loss.item(), 
                        )
                    else:
                        progress_bar.set_postfix(epoch=epoch_num, Loss=loss.item())
                    
                    if self.length_bp_penalty:
                        tbx.add_scalar('train/length_loss', penalty_loss.item(), global_idx)

                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, self.discriminator, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.splits(',')
    dataset_dict = None
    dataset_name = ''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
        if args.synonym and split_name == 'train':
            with open(data_dir + '_eda/' + dataset + '.json', 'rb') as f:
                aug_dict = json.load(f)
                print(f'Augmented {dataset}')
                print(f"Before merging: {len(dataset_dict['question'])}, with aug_dict of length: {len(aug_dict['question'])}")
                dataset_dict = util.merge(dataset_dict, aug_dict)
            dataset_dict = util.merge(dataset_dict, aug_dict)
            print(f"Post merging: {len(dataset_dict['question'])}")

    print("This is the length of the set of dataset dict of id", len(set(dataset_dict['id'])))
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name == 'train')), dataset_dict

def get_clf_dataset(args, in_dataset, in_data_dir, ood_dataset, ood_data_dir, tokenizer):
    in_label_num = None
    ood_label_num = None
    in_dict = util.read_squad(f'{in_data_dir}/{in_dataset}')
    ood_dict = util.read_squad(f'{ood_data_dir}/{ood_dataset}')
    in_label_num = len(list(in_dict['id']))
    ood_label_num = len(list(ood_dict['id']))
    print('in,out:', in_label_num, ood_label_num)

    in_encodings = read_and_process(args, tokenizer, in_dict, in_data_dir, in_dataset, 'eval')
    ood_encodings = read_and_process(args, tokenizer, ood_dict, ood_data_dir, ood_dataset, 'eval')
    print('in,out',len(list(in_encodings['input_ids'])), len(list(ood_encodings['input_ids'])))
    
    return util.CLFDataset(in_encodings, ood_encodings)

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    checkpoint = "./distilbert-base-uncased"

    if args.resume_training:
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    else:
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        trainer = Trainer(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        if args.adv:
            log.info('Preparing ood Training Data...')
            domain_clf_dataset = get_clf_dataset(args, args.train_datasets, args.val_dir, 'race', 'datasets/oodomain_train', tokenizer)
            clf_loader = DataLoader(domain_clf_dataset,
                                    batch_size=2*args.batch_size,
                                    sampler=RandomSampler(domain_clf_dataset)
            )

            best_scores = trainer.train(model, train_loader, val_loader, val_dict, clf_loader)

        else:
            best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_finetune:
        print('Starting outdomain finetune')
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        print(checkpoint_path)
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)

        # This will be code for reinitializing the top num-layer transformer blocks
        for layer in range(args.num_layers):
            # Get top most layers
            layer = 5 - layer
            for module in model.distilbert.transformer.layer[layer].modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        # Standard fine tuning approach of just fine-tuning the output layer after reinitializing.
        # We should probably look to see if we want to change this output layer.
        model.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        model.qa_outputs.bias.data.zero_()

        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(args.device)
        args.num_epochs = args.finetune_epochs
        trainer = Trainer(args, log)
        log.info('Preparing finetuning Data...')
        finetune_train_datasets, _ = get_dataset(args, 'race', 'datasets/oodomain_train', tokenizer, 'train')
        log.info('Preparing finetuning validation Data')
        finetune_val_datasets, finetune_val_dict = get_dataset(args, 'race', 'datasets/oodomain_val', tokenizer, 'validation')
        finetune_train_loader = DataLoader(finetune_train_datasets,
                        batch_size=args.batch_size,
                        sampler=RandomSampler(finetune_train_datasets))
        print("The length of the train loader is : ", len(finetune_train_loader))
        finetune_val_loader = DataLoader(finetune_val_datasets,
                        batch_size=args.batch_size,
                        sampler=SequentialSampler(finetune_val_datasets))
        print(len(finetune_train_loader))
        log.info('Starting Finetuning')
        best_scores = trainer.train(model, finetune_train_loader, finetune_val_loader, finetune_val_dict)
        return best_scores


    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        trainer = Trainer(args, log)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])

if __name__ == '__main__':
    main()