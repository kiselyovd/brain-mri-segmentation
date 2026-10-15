# Component licenses and attribution

The repository's [MIT license](https://github.com/kiselyovd/brain-mri-segmentation/blob/main/LICENSE) covers its original software. It is not a blanket license for training data, upstream weights or generated dataset derivatives.

| Component | Recorded terms | Source |
|---|---|---|
| Original project code | MIT | Repository LICENSE |
| LGG MRI images/masks and source-derived split manifest | CC-BY-NC-SA-4.0; attribution to Mateusz Buda and TCIA/TCGA-LGG | [Original dataset metadata](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/metadata) |
| SegFormer upstream | NVIDIA Source Code License for SegFormer, including research/evaluation noncommercial use limitation | [NVlabs license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |
| NVIDIA SegFormer-B2 ADE20k checkpoint | Upstream model card labels the license `other`; follow its upstream notices | [Base model card](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) |
| Fine-tuned MRI weights | Derived from the above base checkpoint and trained on the above dataset; no independent unrestricted-MIT grant is asserted | Model card and artifact provenance |

The software citation identifies this project. Cite the original dataset and architecture separately when using them. The replay records recover an evaluation protocol; they do not provide clinical validation.
