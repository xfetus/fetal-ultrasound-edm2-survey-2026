# Ultrasound Image Evaluation Survey

A GitHub Pages survey for differentiating real vs. synthetic ultrasound images and rating their clinical quality.

**Dataset:** [harveymannering/ultrasound_images_diffusion](https://huggingface.co/datasets/harveymannering/ultrasound_images_diffusion)

## How the survey works:

100 images are sampled from the 30k dataset using a random seed.
Images are fetched live from the HuggingFace Datasets Server API at runtime — no need to host images yourself
Labels are completely ignored — images are presented unlabelled and randomly shuffled
For each image, raters answer:

Q1: Real or Synthetic (big toggle buttons)
Q2: Quality 1–5 (labelled buttons: Poor → Excellent)

Auto-advances to the next image when both questions are answered
Progress saves to localStorage so page refreshes don't lose work
Export CSV button in the header exports responses at any time
Final download gives a clean CSV: image_number, dataset_index, classification, quality_rating


## Deploy to GitHub Pages

1. Create a new GitHub repository (e.g. `fetal-ultrasound-edm2-survey-2026`)
2. Go to **Settings → Pages → Branch: main → / (root)** → Save
3. Visit `https://xfetus.github.io/fetal-ultrasound-edm2-survey-2026/`



## Export
Responses can be downloaded as **CSV** or **JSON** at the end of the survey. Each row includes:
- Image index, HuggingFace row index
- Ground-truth label (`real` / `synthetic`)
- User's origin guess and quality rating
- Whether the origin was correctly identified


## Notes
* One thing to note: the HuggingFace Datasets API is public and doesn't require authentication, but it does have rate limits. If you have many concurrent raters, images may load a little slowly for the first few.
* The same 100 images are shown to every rater (seeded random selection, seed=20250515)
* Images are fetched live from the HuggingFace Datasets Server API — no images are bundled
* Requires internet access to load images
* Progress is auto-saved to localStorage so the session survives page refreshes

# Scripts

## Installation
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
uv sync
uv pip list --verbose #check versions
source .venv/bin/activate #If needed, to activate the virtual environment
```


## Run scripts
```bash
uv run python scripts/auc.py
```





