#/bin/sh
# declare -a arr=("demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "human_nontata_promoters")
# for dataset in "${arr[@]}"
for dataset in "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "human_nontata_promoters"
do
    echo dataset_$dataset
    papermill run.ipynb experiment_$dataset.ipynb -p dataset_name $dataset
done