# TASI_ERAv2_S19

## Objective

1. Train a mini GPT model following the instruction from Andrej Karpathy in this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s)
2. Upload the model to HuggingFace Apps

## Dataset - tiny shakespeare, character-level
Tiny shakespeare, of the good old char-rnn fame :) Treated on character-level.

- Tokenization performed on Character level
- Vocab size 65. Following are the unique tokens
    - `!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Number of total tokens - 1115394
    - trained on 1,003,854 tokens (90%)
    - validation is performed on 111,540 tokens (10%)
 
## Steps

### Initial experiment

1. Followed the [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) and created the [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S19/blob/main/gpt_dev.ipynb) for experiment.
2. Updated the model code for adding Gradio in the notebook as part of experiment
3. Added the Gradio app in the notebook

### Pushed Model to HuggingFace Model Hub
1. Refactored code in train.py and model.py
2. Added code to same vocab and model arguments and weights that would need to be used for inferencing later
3. Pushed the model.py and saved artifacts to HuggingFace Model Hub using huggingface API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S19/blob/main/gpt_dev_hfhub.ipynb)

### Pushed Gradio App to HuggingFace Spaces
1. Created app.py that can read the model artefacts and vocab artefacts from HuggingFace Model Hub and launch the app
2. Pushed the model.py, app.py and requirements.txt to HuggingFace spaces using huggingface API from this [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S19/blob/main/gpt-dev-spaces.ipynb)

## The Huggingface Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/sayanbanerjee32/nano_text_generator)

The App takes following as input 
1. Seed Text (Prompt) - This provided as input text to the GPT model, based on which it generates further contents. If no data is provided, the only a space (" ") is provided as input
2. Max tokens to generate - This controls the numbers of character tokens it will generate. The default value is 100.
3. Temperature - This accepts value between 0 to 1. Higher value introduces more randomness in the next token generation. Default value is set to 0.7.
4. Select Top N in each step - This is optional field. If no value is provided (or <= 0), all available tokens are considered for next token prediction based on SoftMax probability. However, if a number is set then only that many top characters will be considered for next token prediction.
