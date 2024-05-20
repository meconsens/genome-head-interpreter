# Genome-Head-Interpreter

## Abstract
genome-head-interpreter aims to explain the attention heads of genomic transformer models DNABERT, Enformer, and scGPT. By analyzing attention scores and employing OpenAI's GPT-4 for exploratory prompting, this project aims to elucidate the functional roles of attention heads within these models. Our methods involve Spearman rank correlations to relate attention scores with biological feature annotations, providing new insights into the interpretability of genomic transformers.

## Introduction
Deep learning has shown exceptional capability in extracting meaningful information from genomic sequences. Transformer models, known for their efficiency and scalability, offer promising avenues for genomic interpretation but often act as "black boxes". genome-head-interpreter seeks to shed light on these models by interpreting the roles of their attention heads, linking them to biological features and providing a clearer understanding of their decision-making processes.

## Installation
To set up genome-head-interpreter, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/meconsens/genome-head-interpreter/
cd genome-head-interpreter
```
## Usage
Follow these steps to execute the analysis pipeline for each model you wish to analyze:

1. **Upload Score Matrices**: Place your generated scores matrix into the directory `genome-head-interpreter/preprocessing/data/scores`.

2. **Run Coefficient Collection**: Set up the environment and execute the `collect_coef.py` script. Example for DNABERT:
   ```bash
   export MODEL_NAME=DNABERT
   export DATA_PATH=/genome-head-interpreter/preprocessing/data/scores/${MODEL_NAME}_scores.csv
   export FULL_PATH=/genome-head-interpreter/preprocessing/

   python /genome-head-interpreter/preprocessing/collect_coef.py \
       --model_name $MODEL_NAME \
       --full_path $FULL_PATH \
       --data_path $DATA_PATH
    ```
3. **Generate Random Coefficients**: Shuffle attention scores by running generate_random_coefs.py. Example for DNABERT:
    ```bash
    export MODEL_NAME=DNABERT
    export DATA_PATH=/genome-head-interpreter/preprocessing/data/scores/${MODEL_NAME}_scores.csv
    export FULL_PATH=/genome-head-interpreter/preprocessing/

    python /genome-head-interpreter/preprocessing/generate_random_coefs.py \
        --model_name $MODEL_NAME \
        --full_path $FULL_PATH \
        --data_path $DATA_PATH
    ```
4. **Format Head Ranks**: After generating random distributions, format the heads for prompting by running run_format_head_ranks.py. Example for DNABERT:
    ```bash
    export MODEL_NAME=DNABERT
    export COEF_PATH=/genome-head-interpreter/preprocessing/data/coef/coef_${MODEL_NAME}_results.csv
    export PVAL_PATH=/genome-head-interpreter/preprocessing/data/coef/pval_${MODEL_NAME}_results.csv
    export FULL_PATH=/genome-head-interpreter/preprocessing/

    python /genome-head-interpreter/head-explainer/format_head_ranks.py \
        --model_name $MODEL_NAME \
        --full_path $FULL_PATH \
        --coef_path $COEF_PATH \
        --pval_path $PVAL_PATH
    ```
5. **Write Researcher Explanations**: Add your explanations to genome-head-interpreter/preprocessing/data/researcher_explanations.
6. **Generate Prompts**: Execute prompt.py to generate prompts for the model. Example for DNABERT:
    ```bash
    export MODEL_NAME=DNABERT
    export EXPLANATION_PATH=/genome-head-interpreter/preprocessing/data/explanation_prompts/${MODEL_NAME}.json 
    export RESEARCHER_PATH=/genome-head-interpreter/preprocessing/data/researcher_explanations/${MODEL_NAME}.json
    export FULL_PATH=/genome-head-interpreter/head-explainer/

    python /genome-head-interpreter/head-explainer/prompt.py \
        --model_name $MODEL_NAME \
        --full_path $FULL_PATH \
        --explanation_path $EXPLANATION_PATH \
        --researcher_path $RESEARCHER_PATH

    ```
Note: Repeat these steps for each model (DNABERT, Enformer, scGPT) you wish to analyze, adjusting the MODEL_NAME and paths accordingly.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

Thanks to the developers and contributors of DNABERT, Enformer, and scGPT models.
Appreciation goes to OpenAI for providing GPT-4, which plays a crucial role in our interpretability analysis.

## Contact

For issues, questions, or contributions, please open an issue on the GitHub repository or contact the maintainers directly.
