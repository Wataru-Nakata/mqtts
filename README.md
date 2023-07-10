# mqtts

# installation
```
pip install mqtts-lightning
```

# step1 train_quantizer
```
cd examples/
python train_quantizer.py
```

# step2 preprocess corpus
```
cd examples/
python preprocess.py "data.quantizer_path+=path_to_quantizer_ckpt"
```

# step3 train token decoder
```
cd examples/
python train_token_decoder.py "data.quantizer_path+=path_to_quantizer_ckpt"
```

# to synthesize
TBA

