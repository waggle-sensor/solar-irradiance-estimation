## Training
- Source codes for training the models (FCN, U-Net, DeepLabv4 and Partial Least Squares regression) are provided in the folder. <br/>
- The image readers of the models are modified to use Waggle cloud dataset. <br/>
- You can download the dataset through:
```
from torchvision.datasets.utils import download_and_extract_archive
url = 'https://web.lcrc.anl.gov/public/waggle/datasets/WaggleClouds-0.2.0.tar.gz'
download_and_extract_archive(url, 'download', 'data')
```

## Inference
- Source codes for inference the models based on pixel classification results or probability of pixels how much they can be classified as cloud are provided in the folder. <\br>
- The source codes in `raw` folder is stand alone modules for inference. Additional information how to run the codes will be provided (3/27/2021).
- The modules in `plugins` are in testing to make them as plugins to run them in Waggle nodes (3/27/2021).
