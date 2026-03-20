# Description
This is for PIT: Prompt Invariant Training


## Env Setup (generic stuff)
1. create a venv. For my python version the following works
```python3 -m venv .venv```

2. get in the venv
```source .venv/bin/activate```

3. Install the requirements
```python -m pip install requirements.txt```

4. Put api key from [openrouter](https://openrouter.ai/)
```export DEEP_SEEK_API_KEY=your api key```

## Getting Started
1. Create the dataset folder 
```mkdir dataset``` 

2. Get the gsm8k dataset [link](https://huggingface.co/datasets/openai/gsm8k) in the dataset folder. Running the file `prepare_data.py` might work. 

3. Run make_paraphrase.py with given start point
```
python make_paraphrase.py --start-from 1000 
```

