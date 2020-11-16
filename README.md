# Ordering Sentences and Paragraphs with Pre-trained Encoder-Decoder Transformers and Pointer Ensembles

Code, models and data for the paper *Ordering Sentences and Paragraphs with Pre-trained Encoder-Decoder Transformers and Pointer Ensembles* under review at NAACL2021.

## Getting started

Create the environnement, activate and install requirements.

```bash
conda create -n ordering python=3.7
conda activate ordering
pip install -r requirements.txt
```

## Models weights

All the models weights are available in a [Drive folder](https://drive.google.com/drive/folders/1pSLMX8CLJzoF4rUTSuFd2omPAjLLsBmQ?usp=sharing).
Once the model folder downloaded, put the folder in the ``models/`` folder.

## Datasets

We use the [``datasets``](https://github.com/huggingface/datasets) library from HuggingFace to load and access the datasets.
The dataset custom loading script are in the ``dataset/`` folder.

To load a dataset, run:

```python
from datasets import load_dataset

dataset = load_dataset("path/to/dataset/python/file")
```

To directly access to the data without using ``datasets``, you can find the link to the data in the dataset loading scripts.

## Train a model

We create our models on top of the [``transformers``](https://github.com/huggingface/transformers) library from Huggingface and use the [Trainer](https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/trainer.py) to train our models.
The configuration files for the models presented in the paper are in the ``training/args/`` folder.

To retrain a model, run:

```bash
python run.py --model model --args_file path/to/json/file
```

``model`` is the the model to train (``default`` for BART + simple PtrNet, ``deep`` for BART + deep PtrNet, ``multi`` for BART + multi PtrNet, or ``baseline`` for our baseline LSTM+Attention) and ``path/to/json/file`` is the path to the configuration file to use (note that the configuration file should correspond to ``model``).

To change the training parameters you can directly change the configuration file or create a new one.

## Evaluate the models

To evaluate the models on a dataset, we create configuration files in the ``evaluation/args/``.

To run the evaluation, run:

```python
from evaluation.benchmark import Benchmark
ben = Benchmark.from_json("path/to/json/file")
df = ben.run()
print(df)
>> *dataframe containing the results*
```

## Use the models

Use the ``OrderingModel`` class from ``use.py``. 
For example, to use BART + multi PtrNet trained on the Wikipedia dataset:

```python
from use import OrderingModel
from training.scripts.models.bart_multi import  BartForSequenceOrderingWithMultiPointer

model = OrderingModel(BartForSequenceOrderingWithMultiPointer, "models/bart-base-multi-best-wikipedia", "facebook/bart-base")

PASSAGES_TO_ORDER = ["p3", "p2", "p1", "p4"]

model.order(PASSAGES_TO_ORDER)
>> ["p1", "p2", "p3", "p4"]
```
