# On Sensitivity and Robustness of Normalization Schemes to Input Distribution Shifts in Automatic MR Image Diagnosis

**Authors**: [Divyam Madaan](https://dmadaan.com/),  [Daniel Sodickson](https://med.nyu.edu/faculty/daniel-k-sodickson), [Kyunghyun Cho](https://kyunghyuncho.me), [Sumit Chopra](https://www.spchopra.org)

## Abstract

Magnetic Resonance Imaging (MRI) is considered the gold standard of medical imaging because of the excellent soft-tissue contrast exhibited in the images reconstructed by the MR pipeline, which in-turn enables the human radiologist to easily discern many pathologies. More recently, Deep Learning (DL) models have also achieved state-of-the-art performance in diagnosing multiple diseases using these reconstructed images as input. However, the image reconstruction process within the MR pipeline, which requires the use of complex hardware and adjustment of a large number of scanner parameters, is highly susceptible to noise of various forms resulting in arbitrary artifacts within the images. Furthermore, the noise distribution is not stationary and varies within a machine, across machines, and across patients, leading to varying artifacts within the images.
Unfortunately DL models are quite sensitive to these varying artifacts as it leads to changes in the input data distribution between the training and testing phases.
The lack of robustness of these models against varying artifacts impedes their use in medical applications where safety is critical. In this work, we focus on improving the generalization performance of these models in the presence of multiple varying  artifacts that manifest due to the complexity of the MR data acquisition.
In our experiments we observe that Batch Normalization (BN), a widely used technique during training of DL models for medical image analysis, is a significant cause of performance degradation in these changing environments. As a solution, we propose to use other normalization techniques, such as Group Normalization (GN) and Layer Normalization (LN), to inject robustness into model performance against varying image artifacts.Through a systematic set of experiments, we show that GN and LN provide the better accuracy for a variety of MR artifacts and distribution shifts.

__Contribution of this work__

- We highlight the susceptibility of batch normalization in the task of disease prediction when encountering distribution shifts and various artifacts present in practical clinical scenarios of Magnetic Resonance Imaging (MRI).
- We show that alternate normalization strategies, such as group normalization and layer normalization for intermediate layers, are more robust compared to batch normalization and essential for training deep neural networks that are more robust to these issues.
- We further explore the reasons for the susceptibility of batch normalization.

## Installation and usage

* __Requirements__ Refer to requirements.txt for installing required dependencies.

```
$ pip install -r requirements.txt
```

* __Data Generation__ The fastMRI can be downloaded from [this link](https://fastmri.med.nyu.edu) and the annotations can be found at [this link](https://github.com/microsoft/fastmri-plus/tree/main/Annotations). The original fastMRI data contains volume level slices so generate the Slice level processed data after updating the required paths in the file:

```
$ cd data_processing/knee/
$ python knee_singlecoil.py
```

* Generate the train, validation, and test splits

```
$ cd data_processing/knee/
$ python generate_knee_metadata.py
```

* Train the model using the script

```
$ cd src/
$ python rss_classifier.py
```

The normalization schemes can be changed using the --norm flag and the noise type for evaluation can be changed using the --noise\_type flag.

## Contributing

We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here.

### Citation

If you found the provided code useful, please cite our work.

```bibtex
@inproceedings{madaan2023sensitivity,
  title={On Sensitivity and Robustness of Normalization Schemes to Input Distribution Shifts in Automatic MR Image Diagnosis},
  author={Madaan, Divyam and Sodickson, Daniel and Cho, Kyunghyun and Chopra, Sumit},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}

```
