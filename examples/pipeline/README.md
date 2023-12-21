## build
you may need to build [ppl-cv-related](https://torchpipe.github.io/docs/backend-reference/Ppl.cv) backends:
```bash
BUILD_PPLCV=1 pip install -e .
``` 

## Generate onnx for resnet18 and yolov8

```bash
python3 export_onnx.py 
```

## Execution
Execute `python pipeline.py` to start the inference. Finally, the visualization result ( `dog_result.jpg`) will be generated in the directory.

 