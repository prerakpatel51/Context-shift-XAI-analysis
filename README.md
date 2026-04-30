# When Context Becomes Evidence: Patch-Level XAI for DeiT on NICO++

This project fine-tunes an ImageNet-pretrained DeiT-Small/16 on NICO++ and
audits its patch-level explanations under context shift. The model is trained on
60 object classes across six contexts, then evaluated with transformer,
gradient, perturbation, SHAP, and context-swap explanation methods.


## Main Results

| Metric | Value |
|---|---:|
| Test top-1 accuracy | **0.9047** |
| Macro F1 | **0.9064** |
| Balanced accuracy | **0.9051** |
| Worst-context accuracy | **0.8892** (`dim`) |
| Test loss | 0.4006 |

The aggregate score is strong, but context still matters. Context accuracy spans
from 0.8892 in `dim` to 0.9274 in `grass`, and the report finds that patch-level
context swaps flip 20% of same-class cross-context pairs.

| Context | Accuracy |
|---|---:|
| grass | 0.9274 |
| rock | 0.9198 |
| autumn | 0.9026 |
| water | 0.9016 |
| outdoor | 0.8948 |
| dim | 0.8892 |

<p align="center">
  <img src="readme_assets/accuracy.png" width="31%" alt="Training accuracy curve">
  <img src="readme_assets/macro_f1.png" width="31%" alt="Macro F1 curve">
  <img src="readme_assets/worst_context_accuracy.png" width="31%" alt="Worst-context accuracy curve">
</p>

<p align="center">
  <img src="readme_assets/balanced_accuracy.png" width="31%" alt="Balanced accuracy curve">
  <img src="readme_assets/token_transformation_by_layer.png" width="31%" alt="Token transformation by layer">
</p>

## Explanation Findings

Faithfulness is measured by top-k deletion AUC on 200 stratified samples. Lower
AUC means the method identified patches whose removal most damages the target
prediction.

| Method | Deletion AUC | Std. |
|---|---:|---:|
| **TAM** | **0.1579** | 0.1711 |
| Attention rollout | 0.2022 | 0.1850 |
| Token transformation | 0.2236 | 0.1926 |
| Causal patch impact | 0.2264 | 0.1941 |
| Occlusion | 0.2264 | 0.1941 |
| Gradient x input | 0.2265 | 0.2021 |
| Kernel SHAP | 0.2562 | 0.2091 |

TAM is the best default method in this run. It keeps the multi-layer attention
flow of rollout, but filters attention paths using positive target-class
gradients. That makes the maps more class-specific and more faithful by deletion
AUC. Attention rollout is the strongest lightweight alternative. Occlusion and
causal patch impact are useful intervention sanity checks, but the current
causal implementation falls back to input-space patch replacement, so it behaves
almost exactly like occlusion. SHAP is underpowered at 512 coalitions for a
196-patch ViT grid and produces the weakest deletion score.

<p align="center">
  <img src="readme_assets/deletion_auc_by_mode.png" width="45%" alt="Deletion AUC by method">
  <img src="readme_assets/deletion_curves_top.png" width="45%" alt="Top deletion curves">
</p>

<p align="center">
  <img src="readme_assets/comprehensiveness.png" width="45%" alt="Comprehensiveness by method">
</p>

## XAI Method Math Summary

Let an image $x$ be split into $P = 196$ patches on a $14 \times 14$ grid. Let
$f_y(x)$ be the target-class logit, $p_y(x)$ the target-class probability, and
$s_i$ the saliency score for patch $i$. Every method produces a patch score map
$s \in \mathbb{R}^{14 \times 14}$, then the map is min-max normalized for
visualization.

| Method | Patch score | Interpretation |
|---|---|---|
| Token transformation | $s_i = \sum_l \lVert h_i^{(l)} - h_i^{(l-1)} \rVert_2$ | Measures how much patch token $i$ changes across transformer layers. It shows representational rewriting, not direct class causality. |
| Raw CLS attention | $s_i = \frac{1}{H}\sum_{h=1}^{H} A_{h,\mathrm{CLS},i}^{(L)}$ | Uses final-layer attention from the CLS token to patch $i$. It is easy to inspect but is not class-specific. |
| Attention rollout | $\tilde{A}^{(l)} = \frac{A^{(l)} + I}{2}$, $R^{(l)} = \tilde{A}^{(l)}R^{(l-1)}$, $s_i = R_{\mathrm{CLS},i}^{(L)}$ | Propagates attention through layers so the map reflects multi-layer information flow. |
| TAM | $G^{(l)} = \mathrm{ReLU}\left(\frac{\partial f_y}{\partial A^{(l)}}\right)$, $A_{\mathrm{TAM}}^{(l)} = G^{(l)} \odot A^{(l)}$ | Keeps attention paths that positively support the target class. This is why TAM is more class-specific than rollout. |
| Occlusion | $s_i = p_y(x) - p_y(\mathrm{mask}_i(x))$ | Masks patch $i$ and measures the probability drop. Large positive values mean the patch helped the prediction. |
| Causal patch impact | $s_i = \alpha [p_y(x) - p_y(x_{i \leftarrow b})] + \beta D_{\mathrm{KL}}(p(\cdot \mid x)\,\Vert\,p(\cdot \mid x_{i \leftarrow b}))$ | Intervenes on patch $i$ and combines target-probability drop with output-distribution shift. In this run it uses input-space replacement. |
| Gradient x input | $s_i = \sum_{u \in i}\left|x_u \frac{\partial f_y}{\partial x_u}\right|$ | Local first-order sensitivity of the target logit to pixels in patch $i$. Fast, but can be brittle. |
| Kernel SHAP | $\phi_i = \sum_{S \subseteq P \setminus \{i\}} w(S)\,[f_y(S \cup \{i\}) - f_y(S)]$ | Estimates the average marginal contribution of patch $i$ across visible/hidden patch coalitions. |
| Context swap | $\Delta p_y = p_y(x) - p_y(\mathrm{swap}_{\mathrm{top}\text{-}k}(x, x'))$ | Replaces top-attributed patches with patches from a same-class image in another context to test context dependence. |
| Top-k deletion AUC | $\mathrm{AUC} = \int_0^1 p_y(\mathrm{delete}_{k}(x, s))\,dk$ | Faithfulness metric. Lower AUC means the method found patches whose removal hurts the prediction fastest. |

TAM has the best faithfulness score in this project because it combines two
signals: transformer attention flow and target-class gradient direction. Rollout
can preserve attention paths that are unrelated to the predicted class, while
TAM suppresses paths that do not positively support $f_y(x)$.

## Method Explanation Gallery

The report compares every explanation family on the same six samples: cactus in
`dim`, airplane in `autumn`, crab in `dim`, horse in `water`, fishing rod in
`water`, and train in `autumn`. This makes the visual comparison fair because
each method sees the same low-light, object-centered, small-object, animal,
vehicle, and water-context cases.

### Token Transformation

Token transformation shows where patch representations change most across the
transformer blocks. It captures where the model spends representational capacity,
but it is not class-specific, so object and context processing can both appear.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus token transformation](readme_assets/sample_00010732_token_transformation_overlay.png) | ![Airplane token transformation](readme_assets/sample_00000001_token_transformation_overlay.png) | ![Crab token transformation](readme_assets/sample_00012227_token_transformation_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse token transformation](readme_assets/sample_00079663_token_transformation_overlay.png) | ![Fishing rod token transformation](readme_assets/sample_00076958_token_transformation_overlay.png) | ![Train token transformation](readme_assets/sample_00008127_token_transformation_overlay.png) |

### Raw CLS Attention

Raw final-layer CLS attention is stable and easy to inspect, but it is not a
faithfulness guarantee. It shows where the final CLS token attends, not whether
those patches raise the target-class score.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus raw attention](readme_assets/sample_00010732_last_layer_cls_attention.png) | ![Airplane raw attention](readme_assets/sample_00000001_last_layer_cls_attention.png) | ![Crab raw attention](readme_assets/sample_00012227_last_layer_cls_attention.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse raw attention](readme_assets/sample_00079663_last_layer_cls_attention.png) | ![Fishing rod raw attention](readme_assets/sample_00076958_last_layer_cls_attention.png) | ![Train raw attention](readme_assets/sample_00008127_last_layer_cls_attention.png) |

### Attention Rollout

Attention rollout composes attention through all transformer blocks. It is the
strongest lightweight baseline in the report, but it remains class-agnostic and
can preserve context paths as well as object paths.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus rollout](readme_assets/sample_00010732_rollout_overlay.png) | ![Airplane rollout](readme_assets/sample_00000001_rollout_overlay.png) | ![Crab rollout](readme_assets/sample_00012227_rollout_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse rollout](readme_assets/sample_00079663_rollout_overlay.png) | ![Fishing rod rollout](readme_assets/sample_00076958_rollout_overlay.png) | ![Train rollout](readme_assets/sample_00008127_rollout_overlay.png) |

### Transition Attention Maps

Transition Attention Maps (TAM) are the strongest XAI method in this project.
TAM starts from transformer attention, like rollout, but it keeps only the
attention paths that positively support the target class according to the
target-logit gradient. This matters for NICO++ because object patches and
context patches often coexist in the same image. A raw or rollout attention map
can show where information flows, while TAM is closer to answering which patches
help the model choose this class.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus TAM](readme_assets/sample_00010732_tam_overlay.png) | ![Airplane TAM](readme_assets/sample_00000001_tam_overlay.png) | ![Crab TAM](readme_assets/sample_00012227_tam_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse TAM](readme_assets/sample_00079663_tam_overlay.png) | ![Fishing rod TAM](readme_assets/sample_00076958_tam_overlay.png) | ![Train TAM](readme_assets/sample_00008127_tam_overlay.png) |

TAM is the report's primary explanation method. It produces the lowest deletion
AUC and the most focused maps because it weights attention by positive
target-class gradients.

TAM achieves a deletion AUC of **0.1579**, which is lower than attention rollout
at 0.2022 and much lower than Kernel SHAP at 0.2562. In this deletion test, the
top-ranked patches are removed first; a lower curve means the method found
patches that the prediction actually depends on. TAM's score therefore supports
the visual result: the highlighted regions are not only plausible heatmaps, they
are the patches whose removal most quickly damages the target-class probability.

The sample panels show the same pattern qualitatively. For the cactus, airplane,
crab, and train examples, TAM concentrates most heat on object structure. In the
horse and fishing-rod water examples, TAM still finds the object better than
rollout, but some high-scoring patches remain in the surrounding scene. That is
an important finding rather than a failure of the method: even the most faithful
map shows that the classifier sometimes mixes object evidence with context
evidence.

The cross-context panels make this clearer. For cactus, TAM stays mostly on the
object in autumn, dim, grass, rock, and outdoor settings, but the water-context
case has more background saliency. The same pattern appears for horse, fishing
rod, and train. TAM is therefore useful for two separate jobs: it is the best
method for explaining individual predictions, and it is also a diagnostic for
context leakage when the same object is shown across different environments.

Practical interpretation: use TAM as the first explanation map, but do not stop
there. Compare it with rollout to see how much gradient filtering changes the
attention story, check occlusion or causal patch impact as an intervention
sanity check, and use same-object context panels to decide whether the evidence
is object-centered or scene-dependent.

### Occlusion Sensitivity

Occlusion directly masks each patch and measures how the prediction changes. It
is slower and blockier than attention methods, but it is a useful intervention
sanity check.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus occlusion](readme_assets/sample_00010732_occlusion_overlay.png) | ![Airplane occlusion](readme_assets/sample_00000001_occlusion_overlay.png) | ![Crab occlusion](readme_assets/sample_00012227_occlusion_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse occlusion](readme_assets/sample_00079663_occlusion_overlay.png) | ![Fishing rod occlusion](readme_assets/sample_00076958_occlusion_overlay.png) | ![Train occlusion](readme_assets/sample_00008127_occlusion_overlay.png) |

### Causal Patch Impact

Causal patch impact is intended to measure the effect of patch interventions on
the model output distribution. In this run it falls back to input-space patch
replacement, so the maps are close to occlusion.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus causal impact](readme_assets/sample_00010732_causal_impact_overlay.png) | ![Airplane causal impact](readme_assets/sample_00000001_causal_impact_overlay.png) | ![Crab causal impact](readme_assets/sample_00012227_causal_impact_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse causal impact](readme_assets/sample_00079663_causal_impact_overlay.png) | ![Fishing rod causal impact](readme_assets/sample_00076958_causal_impact_overlay.png) | ![Train causal impact](readme_assets/sample_00008127_causal_impact_overlay.png) |

### Gradient x Input

Gradient x input is fast and class-specific, but the report finds it more
brittle than rollout or TAM, especially in low-light and context-heavy samples.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus gradient x input](readme_assets/sample_00010732_patch_heatmap.png) | ![Airplane gradient x input](readme_assets/sample_00000001_patch_heatmap.png) | ![Crab gradient x input](readme_assets/sample_00012227_patch_heatmap.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse gradient x input](readme_assets/sample_00079663_patch_heatmap.png) | ![Fishing rod gradient x input](readme_assets/sample_00076958_patch_heatmap.png) | ![Train gradient x input](readme_assets/sample_00008127_patch_heatmap.png) |

### Kernel SHAP

Kernel SHAP is model-agnostic, but the tested 512-coalition budget is small for
196 correlated ViT patches. The report finds the maps noisy and least faithful
by deletion AUC.

| Cactus / dim | Airplane / autumn | Crab / dim |
|---|---|---|
| ![Cactus SHAP](readme_assets/sample_00010732_shap_overlay.png) | ![Airplane SHAP](readme_assets/sample_00000001_shap_overlay.png) | ![Crab SHAP](readme_assets/sample_00012227_shap_overlay.png) |

| Horse / water | Fishing rod / water | Train / autumn |
|---|---|---|
| ![Horse SHAP](readme_assets/sample_00079663_shap_overlay.png) | ![Fishing rod SHAP](readme_assets/sample_00076958_shap_overlay.png) | ![Train SHAP](readme_assets/sample_00008127_shap_overlay.png) |

## Context-Swap Results

The context-swap intervention replaces the top 25% attributed patches in a
source image with patches from a same-class image in another context. This tests
whether the highlighted patches carry transferable object evidence or
context-specific evidence.

| Source context | Mean true-class probability drop |
|---|---:|
| **water** | **0.230** |
| outdoor | 0.088 |
| dim | 0.070 |
| rock | 0.051 |
| grass | 0.047 |
| autumn | 0.025 |

Across 200 swap pairs, the mean probability drop is 0.061 and 20% of swapped
pairs change the predicted class. The report interprets water as the strongest
shortcut context in this trained model.

<p align="center">
  <img src="readme_assets/context_swap_drop_by_context.png" width="45%" alt="Context swap probability drop by context">
  <img src="readme_assets/context_swap_probability_drop_hist.png" width="45%" alt="Context swap probability drop histogram">
</p>

The three examples below show the selected context-swap mask, the swapped image,
and the counterfactual perturbation for the same report samples.

| Sample | Context-swap mask | Swapped image | Counterfactual |
|---|---|---|---|
| Cactus / dim | ![Cactus context mask](readme_assets/sample_00010732_context_proxy_overlay.png) | ![Cactus swapped](readme_assets/sample_00010732_context_swapped.png) | ![Cactus counterfactual](readme_assets/sample_00010732_counterfactual.png) |
| Airplane / autumn | ![Airplane context mask](readme_assets/sample_00000001_context_proxy_overlay.png) | ![Airplane swapped](readme_assets/sample_00000001_context_swapped.png) | ![Airplane counterfactual](readme_assets/sample_00000001_counterfactual.png) |
| Crab / dim | ![Crab context mask](readme_assets/sample_00012227_context_proxy_overlay.png) | ![Crab swapped](readme_assets/sample_00012227_context_swapped.png) | ![Crab counterfactual](readme_assets/sample_00012227_counterfactual.png) |

## Stability and Representation Diagnostics

Stability is measured under Gaussian input noise on 20 samples and three seeds.
Attention rollout is very stable, with Spearman rho 0.9965 and top-10% overlap
0.9217. Gradient x input is less stable, with Spearman rho 0.8984 and top-10%
overlap 0.6790.

Representation metrics show that CLS concentration increases with depth: later
blocks focus evidence into fewer patches, while head diversity peaks in middle
layers and collapses closer to the classifier head.

<p align="center">
  <img src="readme_assets/stability_summary.png" width="45%" alt="Explanation stability summary">
  <img src="readme_assets/cls_concentration_by_layer.png" width="45%" alt="CLS concentration by layer">
</p>

<p align="center">
  <img src="readme_assets/stability_distribution.png" width="31%" alt="Explanation stability distribution">
  <img src="readme_assets/entropy_by_layer.png" width="31%" alt="Attention entropy by layer">
  <img src="readme_assets/head_diversity_by_layer.png" width="31%" alt="Head diversity by layer">
</p>

## Same-Object Context Panels

These are the report's cross-context explanation panels. They keep the object
class fixed and change the surrounding context, which makes context reliance
visible. A purely object-centered classifier would keep the highest-evidence
patches on the object across contexts.

### Cactus With TAM

| Autumn | Dim | Grass | Rock | Water | Outdoor |
|---|---|---|---|---|---|
| ![Cactus autumn TAM](readme_assets/sample_00000951_tam_overlay.png) | ![Cactus dim TAM](readme_assets/sample_00010732_tam_overlay.png) | ![Cactus grass TAM](readme_assets/sample_00024386_tam_overlay.png) | ![Cactus rock TAM](readme_assets/sample_00060245_tam_overlay.png) | ![Cactus water TAM](readme_assets/sample_00071905_tam_overlay.png) | ![Cactus outdoor TAM](readme_assets/sample_00043620_tam_overlay.png) |

### Cactus With Attention Rollout

| Autumn | Dim | Grass | Rock | Water | Outdoor |
|---|---|---|---|---|---|
| ![Cactus autumn rollout](readme_assets/sample_00000951_rollout_overlay.png) | ![Cactus dim rollout](readme_assets/sample_00010732_rollout_overlay.png) | ![Cactus grass rollout](readme_assets/sample_00024386_rollout_overlay.png) | ![Cactus rock rollout](readme_assets/sample_00060245_rollout_overlay.png) | ![Cactus water rollout](readme_assets/sample_00071905_rollout_overlay.png) | ![Cactus outdoor rollout](readme_assets/sample_00043620_rollout_overlay.png) |

### Horse With TAM

| Autumn | Outdoor | Rock | Water |
|---|---|---|---|
| ![Horse autumn TAM](readme_assets/sample_00004344_tam_overlay.png) | ![Horse outdoor TAM](readme_assets/sample_00050099_tam_overlay.png) | ![Horse rock TAM](readme_assets/sample_00063825_tam_overlay.png) | ![Horse water TAM](readme_assets/sample_00079663_tam_overlay.png) |

### Fishing Rod With TAM

| Autumn | Grass | Water |
|---|---|---|
| ![Fishing rod autumn TAM](readme_assets/sample_00002828_tam_overlay.png) | ![Fishing rod grass TAM](readme_assets/sample_00028586_tam_overlay.png) | ![Fishing rod water TAM](readme_assets/sample_00076958_tam_overlay.png) |

### Train With TAM

| Autumn | Grass | Rock | Water |
|---|---|---|---|
| ![Train autumn TAM](readme_assets/sample_00008127_tam_overlay.png) | ![Train grass TAM](readme_assets/sample_00040520_tam_overlay.png) | ![Train rock TAM](readme_assets/sample_00068812_tam_overlay.png) | ![Train water TAM](readme_assets/sample_00087551_tam_overlay.png) |

## Repository Layout

```text
nico-vit-xai/
|-- configs/                     # training and XAI YAML configs
|-- data/                        # raw/processed NICO++ and splits
|-- outputs/                     # generated checkpoints, metrics, figures, explanations
|-- readme_assets/               # README figures copied from report outputs
|-- scripts/                     # train, evaluate, plot, XAI, and data utilities
|-- slurm/                       # sbatch jobs for cluster runs
|-- src/
|   |-- datasets/                # NICO++ loader, split utilities, transforms
|   |-- models/                  # timm DeiT builder, hooks, checkpoint helpers
|   |-- training/                # trainer, metrics, logger, losses
|   |-- utils/                   # config, device, paths, IO
|   `-- xai/                     # attribution, interventions, evaluation helpers
|-- environment.yml
|-- requirements.txt
|-- proposal.tex
`-- report.tex
```

## Setup

```bash
conda env create -f environment.yml
conda activate nico-vit-xai
bash scripts/setup_project.sh
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Dataset

Download NICO++ from <https://nico.thumedialab.com/> or the official repository
at <https://github.com/xxgege/NICO-plus>. Place the processed image tree under
`data/processed/nico`, or update `configs/dataset.yaml`.

Supported layouts:

- `root/class_name/context_name/images`
- `root/context_name/class_name/images`
- `root/train/class_name/context_name/images`
- `root/images` with `metadata.csv`

Inspect and split:

```bash
python scripts/inspect_dataset.py --data_root data/processed/nico
python scripts/prepare_nico.py --config configs/dataset.yaml \
  --data_root data/processed/nico \
  --split_dir data/splits
```

Final split sizes from the report: 57,762 train, 13,330 validation, and 17,774
test images. The split CSVs preserve class, context, and class-context group
metadata for every sample.

### Split Creation and Usage

`scripts/prepare_nico.py` first discovers every image and writes three CSV files:
`data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv`.
Each row keeps both the object class and the context:

```text
sample_id,image_path,class_name,context_name,split,class_id,context_id,group_id
sample_00000001,data/processed/nico/autumn/airplane/...,airplane,autumn,test,0,0,airplane__autumn
```

The important fields are:

| Field | Purpose |
|---|---|
| `class_name`, `class_id` | The supervised object label used for model training and classification metrics. |
| `context_name`, `context_id` | The annotated environment/background used for context-wise robustness metrics and context-shift analysis. |
| `group_id` | The combined class-context cell, for example `airplane__autumn`; used to report fine-grained group accuracy. |
| `split` | Marks whether the row belongs to train, validation, or test. |

The default split script uses the configured fractions (`test_fraction: 0.20`,
`val_fraction: 0.15`, seed 42). If metadata already provides a split column, that
split is respected. Otherwise, the script creates train/validation/test splits
with class stratification and preserves the context labels in every row. The
split summary also records context distributions and class-context group counts,
so the run can check whether rare class-context cells are represented.

How the splits are used:

| Stage | Split file | How class/context fields are used |
|---|---|---|
| Training | `train.csv` | The model optimizes cross-entropy on `class_id`. Context is loaded but not used as a target, so the classifier learns object labels while still seeing natural context correlations. |
| Validation | `val.csv` | Used for checkpoint selection, including best validation accuracy, macro-F1, loss, and worst-context accuracy. Context fields support robustness tracking during model selection. |
| Testing | `test.csv` | Used only after training for final metrics. Evaluation reports overall accuracy, macro-F1, balanced accuracy, per-context accuracy from `context_id`, and per-group accuracy from `group_id`. |
| XAI sample selection | `test.csv` plus predictions | `scripts/select_xai_samples.py` evaluates the checkpoint on the test split, then selects examples by confidence/correctness, class balance, and same-class/different-context coverage. |
| XAI methods | selected test rows | Attribution methods use `image_path` and `class_id` to explain the target prediction. Context labels are used to compare explanations across environments and to build context-swap pairs. |
| Context swap | selected same-class pairs | Pairs share the same `class_name` but have different `context_name`; top-attributed patches are swapped to measure whether the evidence is object-centered or context-dependent. |

This is why the project keeps class and context separate: class is the prediction
target, while context is the audit axis used to expose shortcut behavior.

## DeiT Architecture

The model is `deit_small_patch16_224` initialized with ImageNet-1k `fb_in1k`
weights from `timm`. The ImageNet classification head is replaced with a 60-way
linear head for NICO++. All XAI maps are projected back to the 14 x 14 patch grid
formed by the 16 x 16 image patches.

![DeiT-Small architecture diagram](readme_assets/deit_architecture.svg)

The Mermaid source for this diagram is saved at
[readme_assets/deit_architecture.mmd](readme_assets/deit_architecture.mmd).

## Training

```bash
python scripts/train.py \
  --config configs/train_deit_pretrained.yaml \
  --data_root data/processed/nico \
  --split_dir data/splits \
  --output_dir outputs
```

Training uses `deit_small_patch16_224` with ImageNet-1k `fb_in1k` weights,
AdamW, backbone LR `3e-5`, head LR `1e-4`, weight decay `0.05`, cosine decay,
two warm-up epochs, batch size 64, RandAugment, random erasing, and 30 epochs.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt \
  --config configs/train_deit_pretrained.yaml \
  --data_root data/processed/nico \
  --split_dir data/splits \
  --output_dir outputs/metrics/deit_pretrained
```

Plot training curves:

```bash
python scripts/plot_training_metrics.py
```

## XAI Workflow

Select the 200-sample stratified XAI subset:

```bash
python scripts/select_xai_samples.py \
  --checkpoints outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt \
  --names pretrained_best \
  --data_root data/processed/nico \
  --split_dir data/splits \
  --output_csv outputs/metrics/xai/xai_selected_samples.csv \
  --num_samples 200
```

Run the explanation methods:

```bash
python scripts/run_xai.py \
  --checkpoint outputs/checkpoints/deit_pretrained/best_val_macro_f1.pt \
  --checkpoint_name pretrained_best \
  --config configs/xai.yaml \
  --selected_samples outputs/metrics/xai/xai_selected_samples.csv \
  --data_root data/processed/nico \
  --split_dir data/splits \
  --output_dir outputs/explanations/pretrained_best
```

Aggregate plots and summaries:

```bash
python scripts/plot_xai_results.py \
  --explanation_dirs outputs/explanations/pretrained_best \
  --names pretrained_best \
  --selected_samples outputs/metrics/xai/xai_selected_samples.csv \
  --output_dir outputs/figures/xai
```

Implemented methods include token transformation, gradient x input, occlusion,
attention rollout, raw attention visualization, causal patch impact, Transition
Attention Maps, Kernel SHAP, context swap, counterfactual perturbation, CLIP
concept dissection, patch-interaction graphs, top-k deletion faithfulness, and
method agreement.

## SLURM

```bash
sbatch slurm/train_pretrained.sbatch
sbatch slurm/select_xai_samples.sbatch
sbatch slurm/run_xai_pretrained_best.sbatch
sbatch slurm/run_xai_token_context.sbatch
sbatch slurm/run_xai_concept_dissect.sbatch
sbatch slurm/plot_xai_results.sbatch
sbatch slurm/run_full_pretrained_xai_pipeline.sbatch
```

## Output Map

| Path | Contents |
|---|---|
| `outputs/checkpoints/deit_pretrained/` | per-epoch, final, last, and best checkpoints |
| `outputs/metrics/deit_pretrained/test_metrics.json` | test accuracy, F1, balanced accuracy, class/context/group metrics |
| `outputs/metrics/deit_pretrained/context_accuracy.csv` | context-wise accuracy |
| `outputs/metrics/xai/xai_selected_samples.csv` | 200 stratified XAI samples |
| `outputs/explanations/pretrained_best/<method>/` | per-sample maps, overlays, and CSVs |
| `outputs/metrics/xai/evaluation/` | faithfulness, stability, and representation summaries |
| `outputs/figures/` | generated plots used in the report |
| `readme_assets/` | selected README-safe copies of report figures |

## Limitations

NICO++ has no pixel-level object masks, so object-versus-background attribution
is estimated qualitatively and through context swaps rather than segmentation
metrics. Causal patch impact falls back to input-space ablation in this run,
making it redundant with occlusion. SHAP uses only 512 coalitions, which is small
for 196 dependent patch features. Results are for one DeiT-Small/16 checkpoint;
larger ViTs or context-balanced training may change the ranking.

## Conclusion

The fine-tuned DeiT-Small/16 performs well on NICO++, reaching 90.47% test
accuracy and 0.9064 macro F1, but the XAI results show that high aggregate
accuracy does not mean the model is context-invariant. The worst context is
`dim` at 88.92%, and patch-level context swaps change 20% of predictions. The
model has learned real object evidence, but it still uses scene evidence in
measurable cases.

TAM is the best explanation method in this project. It gives the lowest deletion
AUC, produces the cleanest object-focused maps, and exposes context leakage more
clearly than raw attention or rollout. The strongest practical workflow is:
evaluate performance by context, generate TAM as the primary explanation, verify
it against rollout and perturbation methods, and then inspect same-object
cross-context panels before claiming that the model reasons from object features
instead of background shortcuts.

The main takeaway is that context dependence is not all-or-nothing. Many
predictions are mostly object-centered, but water and other distinctive
backgrounds can shift evidence away from the object. A deployment audit should
therefore combine accuracy, per-context metrics, faithfulness tests, and
cross-context visual explanations.

## Citation

If you use this pipeline, cite the upstream papers listed in `report.tex`,
including ViT, DeiT, NICO++, attention rollout, transformer interpretability,
TAM, ViT-CX, SHAP, Grad-CAM, and shortcut-learning work.
