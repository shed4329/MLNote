# Project 1 : AIGC detector

## About This project

This project is a simple practise after learning Andrew NG's Machine Learning (ML) course.

> ATTENTION: The result of this programme is just for reference. It's not very reliable.

### Technical Theory Behind
 
The AIGC detector is based on the simplest classification algorithm —— logistic regression. It uses `sigmod` as the activation function.
#### Detection Metrics Introduction
- **avg_sentence_len**: Represents the average length of sentences in the text. It gives an idea of the typical number of elements (like words, tokens) that make up a sentence on average. For example, if most sentences in a text are short, this value will be relatively low, and vice versa.
- **connective_ratio**: The proportion of connective words (such as "and", "but", "however") in the text relative to the total number of words or tokens. Connective words help in understanding the logical flow and relationships between different parts of the text.
- **connective_starter_ratio**: Denotes the proportion of sentences that start with a connective word. It can reflect the writing style regarding how often the author begins a new thought or statement with a connecting term.
- **max_sentence_len**: Shows the length of the longest sentence in the text. This metric helps identify extreme cases of very long, potentially complex or rambling sentences.
- **min_sentence_len**: Indicates the length of the shortest sentence. It can highlight the presence of very concise expressions in the text.
- **passive_voice_ratio**: The proportion of sentences that use the passive voice. Passive voice construction (e.g., "The ball was thrown by the boy") has a different stylistic feel compared to active voice and can vary between AI-generated and human-written texts.
- **sentence_len_std**: The standard deviation of sentence lengths. It measures the variability or dispersion of sentence lengths in the text. A low standard deviation means sentence lengths are relatively consistent, while a high value indicates more variation.
- **special_punctuation**: The proportion or count (depending on implementation) of special punctuation marks (like em dashes, ellipses, etc.) in the text. These can contribute to the overall tone and style.
- **ttr (Type-Token Ratio)**: Calculated as the number of distinct words (types) divided by the total number of words (tokens) in the text. It reflects the lexical richness or vocabulary diversity. A higher TTR means a more diverse use of vocabulary.

## File Description
```
|-dataset
|   |-processed
|   |    |-AIGC(foleder for AIGC files after pre process)
|   |    |-human_being(foleder for human written files after pre process)
|   |-raw
|        |-AIGC(put AIGC essay here, txt format plaease)
|        |-human_being(put human_being essay here, txt format plaease)
|-model(folers for putting model)
|   |-report.txt(report for training process)
|   |-model.json(file storaging model parameters)
|   |-scaler.pkl(file for deal the scale of inputs)
|-src
|   |-IOTest.py(test for reading file)
|   |-model.py(file for training model)
|   |-paraCounter.py(file for satisticing features of paragraphs)
|   |-paraCounterTest.py(file for testing paraCounter)
|   |-preProcess.py(file for processing)
|-main_consle.py(programme in console)
|-main_GUI.py(programmer with graphic user-interface)
|-readme.md(giving information)
```
## Usage

If you want to train your own model, you’re supposed to follow the following steps. If you have already trained model, put your model files at `./model` folder and jump to **run** section.

### pre-process

This process is convert you file into `json` file containing statics of essays.

>ATTENTION: your essays should be **txt** files and put 1 paragraph in 1 line

First, put AIGC essays at `./dataset/raw/AIGC` folder and human written essays at `./dataset/raw/human_being` folder.

Second, run the pre-process code `preProcess.py` at `./src`. Later, marked `json` file will appear at `./dataset/processed/AIGC` and `./dataset/processed/human_being` both.

### training model

Run `model.py` at `./src` . It will automatically run the training process. The training report will be print to your terminal and saved in file `./model/report.txt`. The model files, `./model/model.json` and `./model/scaler.pkl` will be saved automatically.

### run

There are two opinions for you.

If you prefer to run in terminal without GUI, just run `main_console.py`

If GUI may fit you well, just run `main_GUI.py`

## TODO
- Add square and cubic terms of relevant metrics as parameters to address potential underfitting. 
- Also, incorporate a penalty mechanism (such as L1 or L2 regularization in logistic regression) to prevent overfitting, making the model more generalizable and robust.
 