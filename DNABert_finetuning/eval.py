from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score
from genomic_benchmarks.data_check.info import labels_in_order
from tqdm import tqdm

def get_test_score(encoded_samples_test, model, dataset_name):
    print('Computing test score')
    test_loader = DataLoader(
                encoded_samples_test, 
                sampler = SequentialSampler(encoded_samples_test), 
                batch_size = 4 #TODO increase with your CPU
            )

    predictions = []
    # for sample in tqdm(test_loader, total=len(test_dset)/32):

    for sample in tqdm(test_loader, total=len(test_loader)): 

        outputs = model.to("cpu")(**sample)
        # outputs = model(**sample) #TODO make eval on GPU

        preds = outputs.logits.argmax(-1).tolist()
        predictions.extend(preds)

    labels = labels_in_order(dset_name=dataset_name)
    score = f1_score(labels, predictions)
    print('test f1 score is', score)
    return score