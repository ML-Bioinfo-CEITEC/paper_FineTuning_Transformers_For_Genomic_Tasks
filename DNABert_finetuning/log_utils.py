import comet_ml
def log_extra(config, f1_score):
    current_experiment = comet_ml.get_global_experiment()
    afterlog_experiment = comet_ml.ExistingExperiment(previous_experiment=current_experiment.get_key())
    exp_name = f"{config['dataset_name']}:{config['kmer_len']}:{config['stride']}:freeze={config['freeze']}:LR={config['learning_rate']}:WD={config['weight_decay']}:BS={config['batch_size']}:rand_weights={config['random_weights']}:"
    afterlog_experiment.set_name(exp_name)
    afterlog_experiment.log_parameters(config)
    afterlog_experiment.log_metric("test F1 score", f1_score)
    afterlog_experiment.end()

