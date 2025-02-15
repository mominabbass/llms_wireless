# LLMs for Wireless Communications

This codebase is compatible with GPT-J, GPT-2, Llama-2 and any other language model available in [HuggingFace Transformers](https://huggingface.co/models).


## Dependencies

The code is implemented using PyTorch and the [HuggingFace's Transformer repository](https://github.com/huggingface/pytorch-transformers). If you intend to run the code on a local model like GPT-2, it necessitates a single GPU.

## Installation
To setup the anaconda environment, simply run the following command:
```
conda env create -f setup_environment.yaml
```

After installation is complete, run:
```
conda activate fewshot_icl
```

## Datasets
We provide evaluation support for the system demodulation task. You have the flexibility to incorporate additional datasets by defining the prompt format and label space in a manner similar to the existing datasets in data_utils.py.

## Evaluation
You can replicate the results in our paper by running the ssh scripts in the `cls_sh` folder. For example, to run 16-shot on GPT-J, run: `sh cls_sh/cls_gptj_16shot.sh`. Alternatively, copy and paste the contents of the .sh file into the terminal as follows:

```
python run_classification.py --model="gptj" --all_shots="16" --subsample_test_set=300 --epochs=15 --lr=0.0015

```

To execute different experiments, specify the desired arguments in the above command from the corresponding .sh file.

