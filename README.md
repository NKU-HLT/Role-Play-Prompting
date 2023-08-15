# Better Zero-Shot Reasoning with Role-Play Prompting

This is the code for our paper "Better Zero-Shot Reasoning with Role-Play Prompting".

## Environment
```
openai 0.27.8
torch 2.0.1
pandas 2.0.3
```

## Usage 

First, you need an OpenAI API key and write it into utils.py.
```python
# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(1)

    # enter your OpenAI key below
    openai.api_key = ""
```
Then you can run the code using run.sh.
```
bash run.sh
```
More details are presented in run.sh.

We provide our experimental results in my_log folder. And we observe that the results sometimes may not be completely fixed though the temperature is set to 0. But this does not affect our conclusion. See [related discussions](https://community.openai.com/t/a-question-on-determinism/8185/1) in OpenAI. 