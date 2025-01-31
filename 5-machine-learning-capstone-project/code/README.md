# <font color="turquoise"> <p style="text-align:center"> Translating Math Formula Images To LaTeX Sequences </p> </font>



## Setup

- Python >= 3.8.

- Install by conda:
    ```bash
    conda create --name img2latex python=3.8 -y
    conda activate img2latex
    pip install -r requirements.txt
    ```

- Install Without Conda:
    ```bash
    pip install -r requirements.txt
    ```
    

## Uses


#### Available Model Checkpoint
Checkpoint model is available at Huggingface, which can be downloaded at [hoang-quoc-trung/sumen-base](https://huggingface.co/hoang-quoc-trung/sumen-base).


#### Training
```bash
python train.py --config_path src/config/base_config.yaml --resume_from_checkpoint true

arguments:
    -h, --help                   Show this help message and exit
    --config_path                Path to configuration file
    --resume_from_checkpoint     Continue training from saved checkpoint (true/false)
```

#### Inference
```bash
python inference.py --input_image assets/example_1.png --ckpt hoang-quoc-trung/sumen-base

arguments:
    -h, --help                   Show this help message and exit
    --input_image                Path to image file
    --ckpt                       Path to the checkpoint model
```

#### Test
```bash
python test.py --config_path src/config/base_config.yaml --ckpt hoang-quoc-trung/sumen-base

arguments:
    -h, --help                   Show this help message and exit
    --config_path                Path to configuration file
    --ckpt                       Path to the checkpoint model
```

#### Web Demo
```bash
streamlit run streamlit_app.py -- --ckpt hoang-quoc-trung/sumen-base

arguments:
    -h, --help                   Show this help message and exit
    --ckpt                       Path to the checkpoint model
```

## Dataset

Dataset is available here: [Fusion Image To Latex Datasets](https://huggingface.co/datasets/hoang-quoc-trung/fusion-image-to-latex-datasets)

The directory data structure can look as follows:
* Save all images in a folder, replace the path as `root` in config file.
* Prepare a CSV file with 2 columns:
    * `image_filename`: The name of image file.
    * `latex`: Latex code.

Samples:
|image_filename|latex|
|:--:|:--:|
|200922-1017-140.bmp|\sqrt { \frac { c } { N } }
78cd39ce-71fc-4c86-838a-defa185e0020.jpg|\lim_{w\to1}\cos{w}
KME2G3_19_sub_30.bmp|\sum _ { i = 2 n + 3 m } ^ { 1 0 } i x
1d801f89870fb81_basic.png|\sqrt { \varepsilon _ { \mathrm { L J } } / m \sigma ^ { 2 } }