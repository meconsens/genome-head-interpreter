#!/bin/bash

# Directory settings
BASE_DIR="/home/mica/nucleotide-transformer"
SCRIPT_PATH="${BASE_DIR}/genome-interpreter/nucleotide_ablations.py"
echo "Using script: ${SCRIPT_PATH}"

# Activate conda environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nt

echo "Using script: ${SCRIPT_PATH}"

# Run TATA-kmer experiment
echo "========================================================="
echo "Running TATA-kmer experiment"
echo "========================================================="
python "${SCRIPT_PATH}" \
    --model_path "${BASE_DIR}/CUSTOM-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-1000" \
    --data_file "${BASE_DIR}/data/custom_TATA_test.fa" \
    --results_dir "${BASE_DIR}/genome-interpreter/ablation_results/TATA-kmer" \
    --important_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/TATA-kmer/important_heads.py" \
    --unimportant_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/TATA-kmer/unimportant_heads.py" \
    --experiment_name "TATA-kmer"

# #Run TSS experiment
# echo "========================================================="
# echo "Running TSS experiment"
# echo "========================================================="
# python "${SCRIPT_PATH}" \
#     --model_path "${BASE_DIR}/CUSTOM-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-1000" \
#     --data_file "${BASE_DIR}/data/custom_TATA_test.fa" \
#     --results_dir "${BASE_DIR}/genome-interpreter/ablation_results/custom_TATA" \
#     --important_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/custom_TATA/important_heads.py" \
#     --unimportant_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/custom_TATA/unimportant_heads.py" \
#     --experiment_name "TSS"


# #Run Fake TATA experiment
# echo "========================================================="
# echo "Running Fake TATA experiment"
# echo "========================================================="
# python "${SCRIPT_PATH}" \
#     --model_path "${BASE_DIR}/FAKE-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-1000" \
#     --data_file "${BASE_DIR}/data/fake_TATA_test.fa" \
#     --results_dir "${BASE_DIR}/genome-interpreter/ablation_results/fake_TATA" \
#     --important_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/fake_TATA/important_heads.py" \
#     --unimportant_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/fake_TATA/unimportant_heads.py" \
#     --experiment_name "GC"

# #Run Fake TATA kmer experiment
echo "========================================================="
echo "Running Fake TATA k-mer experiment"
echo "========================================================="
python "${SCRIPT_PATH}" \
    --model_path "${BASE_DIR}/FAKE-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-1000" \
    --data_file "${BASE_DIR}/data/fake_TATA_test.fa" \
    --results_dir "${BASE_DIR}/genome-interpreter/ablation_results/fake_TATA-kmer/" \
    --important_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/fake_TATA-kmer/important_heads.py" \
    --unimportant_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/fake_TATA-kmer/unimportant_heads.py" \
    --experiment_name "TATA-kmer"


# Run GC experiment (enhancer)
# echo "========================================================="
# echo "Running GC experiment (enhancer)"
# echo "========================================================="
# python "${SCRIPT_PATH}" \
#     --model_path "${BASE_DIR}/ENHANCER-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-4000" \
#     --data_file "${BASE_DIR}/data/enhancer_test.fa" \
#     --results_dir "${BASE_DIR}/genome-interpreter/ablation_results/enhancer" \
#     --important_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/enhancer/important_heads.py" \
#     --unimportant_heads_file "${BASE_DIR}/genome-interpreter/ablation_data/enhancer/unimportant_heads.py" \
#     --experiment_name "GC"

echo "All experiments completed!"