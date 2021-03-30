## Training
- Source codes for training the models (FCN, U-Net, DeepLabv4 and Partial Least Squares regression) are provided in the folder. <br/>
- `main.py` in fcn, unet and deeplabv3 is the main function to train the models.
- The image readers of the models are modified to use Waggle cloud dataset. <br/>
- You can download the dataset through:
```
from torchvision.datasets.utils import download_and_extract_archive
url = 'https://web.lcrc.anl.gov/public/waggle/datasets/WaggleClouds-0.2.0.tar.gz'
download_and_extract_archive(url, 'download', 'data')
```

## Inference
- Source codes for inference the models based on pixel classification results or probability of pixels how much they can be classified as cloud are provided in the folder. <\br>
- For detail information, see
```
@article{park2021prediction,
  title={Prediction of Solar Irradiance and Photovoltaic Solar Energy Product Based on Cloud Coverage Estimation Using Machine Learning Methods},
  author={Park, Seongha and Kim, Yongho and Ferrier, Nicola J and Collis, Scott M and Sankaran, Rajesh and Beckman, Pete H},
  journal={Atmosphere},
  volume={12},
  number={3},
  pages={395},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
- The source codes in `raw` folder are stand alone modules for inference. Each `run.sh` shows how to run the codes.
- The modules in `plugins` are in testing to make them as plugins to run them in Waggle nodes (3/27/2021).

## Notes to developers:
- Timestamp for cloud cover needs to be the time when the images captured: The timestamp must be provided with the images.
- The waggle.plugin.publish function is tested, and checked with [log](https://github.com/waggle-sensor/pywaggle/wiki/Plugins:-Getting-Started#debug-logging).
- Not yet plugin-ized (3/30/2021): needs to be dockerized and create sage.json and others using [virtual waggle](https://github.com/waggle-sensor/virtual-waggle#running-node-application-stack)
