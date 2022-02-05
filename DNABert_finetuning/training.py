from models import get_dnabert
from preprocessing import get_preprocessed_datasets
from transformers import Trainer, EarlyStoppingCallback
from transformers.integrations import CometCallback


def metrics(pred) -> dict:
    acc = sum([pred == true for pred, true in zip(pred.predictions.argmax(-1).tolist(), 
                                                  pred.label_ids.tolist())]) / len(pred.label_ids)
    return {"acc": acc}

def get_trained_model(config, args, model, tokenizer):

    encoded_samples, encoded_samples_test = get_preprocessed_datasets(
        config['dataset_name'], 
        tokenizer, 
        kmer_len=config['kmer_len'], 
        stride=config['stride'],
    )

    ratio = config['eval_dset_ratio']
    train_dset_len = len(encoded_samples)
    val_dset_len = int(train_dset_len*ratio)
    print('Dataset', config['dataset_name'], 'of length', train_dset_len, 'has valid size of', val_dset_len, 'that is ratio of ', ratio)
    

    trainer = Trainer(model=model, args=args, compute_metrics=metrics, callbacks=[EarlyStoppingCallback(config['early_stopping_patience']), CometCallback()],
                    train_dataset=encoded_samples[:-val_dset_len], eval_dataset=encoded_samples[-val_dset_len:])
    trainer.train() 

    #TODO get_test_set_differently?
    return trainer.model, encoded_samples_test