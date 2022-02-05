import torch
import random
from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset


def preprocess_datasets(train_dset, test_dset, tokenizer, kmer_len, stride):
    '''
    Currently 
    AABBC with stride = 2 and kmer = 2 produces
    ['AA','BB'] and cuts off C 
    TODO is this desired? If not, set for cycle limit to len(text) only
    '''

    offset = kmer_len

    #This is where the max_length is computed
    max_seq_len = max([len(train_dset[i][0]) for i in range(len(train_dset))])
    max_seq_len = max_seq_len if max_seq_len < 512 else 512 

    if(max_seq_len >=512):
        print('WARNING: some sequences are longer than 512, these are being trimmed')

    def coll(data):
        encoded_dset = [(label, tokenizer([text[i:i+offset] 
                        for i in range(0, len(text)-offset+1, stride)], max_length=max_seq_len, padding="max_length", is_split_into_words=True, truncation=True, verbose=True).input_ids)
                        for text, label in data]
        encoded_samples = [{"input_ids": torch.tensor(ids), "attention_mask": torch.tensor([1]*len(ids)), "labels": torch.tensor(label)} 
                    for label, ids in encoded_dset]

        return encoded_samples

    encoded_samples = coll(train_dset)
    encoded_samples_test = coll(test_dset)

    random.shuffle(encoded_samples) #Shuffle the train set

    return encoded_samples, encoded_samples_test

def get_preprocessed_datasets(dataset_name, tokenizer, kmer_len, stride):
    train_dset = get_dataset(dataset_name ,'train', version=0)
    test_dset = get_dataset(dataset_name ,'test', version=0)

    return preprocess_datasets(train_dset, test_dset, tokenizer, kmer_len, stride)

