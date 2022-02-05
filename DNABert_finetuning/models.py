from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def get_cdnabert(config):
    kmer_len=config['kmer_len']
    tokenizer = AutoTokenizer.from_pretrained(f"Vlasta/CDNA_bert_{kmer_len}")
    model = AutoModelForSequenceClassification.from_pretrained(f"Vlasta/CDNA_bert_{kmer_len}")
    model.to('cuda')
    return model, tokenizer


def get_dnabert(config, move_to_cuda=True, verbose=False):
    kmer_len=config['kmer_len']
    freeze=config['freeze']
    layers_to_unfreeze=config['layers_to_unfreeze'] 
    random_weights=config['random_weights']

    if kmer_len not in [3,4,5,6]:
        raise Exception('DNA bert only supports kmer_len 3,4,5, or 6')

    tokenizer = AutoTokenizer.from_pretrained(f"armheb/DNA_bert_{kmer_len}")
    model = AutoModelForSequenceClassification.from_pretrained(f"armheb/DNA_bert_{kmer_len}")
    if(move_to_cuda):
        model.to('cuda')
        print('model device', model.device)

    if(random_weights):
        model.init_weights() #TODO check if inits all layers?
        print('reseting weights')
        # print(list(model.parameters())[0])
    if(freeze):
        print('freezing')
        for p in list(model.parameters())[:-layers_to_unfreeze]: #Un-freezing from the end
            p.requires_grad=False

    if(verbose):
        for p in list(model.parameters()):
            print(p.requires_grad)
        
        print(model.parameters)
    
    return model, tokenizer


