# SegEarth-R1: Geospatial Pixel Reasoning via Large Language Model

<font size=4><div align='center' > [[🤗 Project](https://earth-insights.github.io/SegEarth-R1)] [[🤗 Paper](https://arxiv.org/abs/2504.09644)] [[🤗 Dataset](https://huggingface.co/datasets/earth-insights/EarthReason)] [[🤗 Checkpoint](https://huggingface.co/earth-insights)] </div></font>

<div align="center">
<div>
  <font size=4>
      <p>🎉<b>TL;DR</b> We introduce the geospatial pixel reasoning task, construct the first benchmark dataset (EarthReason), and propose a simple yet effective baseline (SegEarth-R1).</p>
  </font>
</div>
<img src="https://github.com/user-attachments/assets/01b7ac65-ecd6-4590-b538-125d6b719bb1" width="100%"/>
</div>

Remote sensing has become critical for understanding environmental dynamics, urban planning, and disaster management. However, traditional remote sensing workflows often rely on explicit segmentation or detection methods, which struggle to handle complex, implicit queries that require reasoning over spatial context, domain knowledge, and implicit user intent. Motivated by this, we introduce a new task, i.e., geospatial pixel reasoning, which allows implicit querying and reasoning and generates the mask of the target region. To advance this task, we construct and release the first large-scale benchmark dataset called EarthReason, which comprises 5,434 manually annotated image masks with over 30,000 implicit question-answer pairs. Moreover, we propose SegEarth-R1, a simple yet effective language-guided segmentation baseline that integrates a hierarchical visual encoder, a large language model (LLM) for instruction parsing, and a tailored mask generator for spatial correlation. The design of SegEarth-R1 incorporates domain-specific adaptations, including aggressive visual token compression to handle ultra-high-resolution remote sensing images, a description projection module to fuse language and multi-scale features, and a streamlined mask prediction pipeline that directly queries description embeddings. Extensive experiments demonstrate that SegEarth-R1 achieves state-of-the-art performance on both reasoning and referring segmentation tasks, significantly outperforming traditional and LLM-based segmentation methods.

## 🗞️ Update

- **`2025-05-04`**: The code, dataset and checkpoints are released.
- **`2025-04-15`**: 🔥🔥🔥 We release the paper of SegEarth-R1 on [arXiv](https://arxiv.org/abs/2504.09644).

## 🔧 Usage：

Follow the guidelines below to set up, train and evaluate:

* [ Preparation ⚙️](docs/Preparation.md): Instructions for organizing datasets and pretrained weights for proper model training and inference.
* [Installation 💻](docs/Installation.md): Set up the `segearthr1` conda environment, install dependencies, and clone the repo.
* [Training 🏋️‍♂️](docs/Training.md): Run `scripts/train.sh` with DeepSpeed, modifying parameters like data and model paths for training.
* [Evaluation 🎯](docs/Evaluation.md): Run `scripts/eval.sh` to evaluate the model, updating paths as needed. 

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@article{li2025segearth,
  title={SegEarth-R1: Geospatial Pixel Reasoning via Large Language Model},
  author={Li, Kaiyu and Xin, Zepeng and Pang, Li and Pang, Chao and Deng, Yupeng and Yao, Jing and Xia, Guisong and Meng, Deyu and Wang, Zhi and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2504.09644},
  year={2025}
}
```

## 🙏 Acknowledgement

We appreciate [PSALM](https://github.com/zamling/PSALM) and [Mask2Former](https://github.com/facebookresearch/Mask2Former) for making their models and code available as open-source contributions.
