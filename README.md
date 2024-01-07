# VelocityTransformer
# Datasets
```
Datasets
├── NonReset
│   ├── 0001
│   │   ├── control
│   │   │     │ 
│   │   │     ├── 00001.npy
│   │   │     ├── 00002.npy
│   │   │     ├── 00003.npy
│   │   │     ├── ...
│   │   │     
│   │   ├── objects
│   │   ├── path
│   │   ├── pose
│   │   └── trajectory
│   │          │ 
│   │          ├── 00001.npy
│   │          ├── 00002.npy
│   │          ├── 00003.npy
│   │          ├── ...
│   │   
│   ├── 0002
│   ├── 0003
│   ├── 0004
│   ├── ...
```

# Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```
# Train
```
python scripts/train.py 
```
# Predict
```
python scripts/predict.py ckpt={/path/to/trained_ckpt}
```

# Export ONNX
```
python scripts/export_onnx.py ckpt={/path/to/trained_ckpt}
```
