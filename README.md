# Tangle

Tangle is a desktop application that protects original digital artwork from being scraped and used to train generative AI models. It works by applying an invisible mathematical perturbation to an image, imperceptible to the human eye, but destructive to how AI models perceive and learn from it.

---

## Protection Modes

**FGSM** — Fast Gradient Sign Method. Single step adversarial attack, instant results.

**PGD** — Projected Gradient Descent. Iterative attack with random start, significantly stronger protection.

**CLIP** — Style embedding corruption. Targets OpenAI CLIP directly, the model used in most modern style-learning AI. Pushes the artwork's style representation toward a decoy style so any model trained on it learns the wrong artistic identity.

---

## Installation

    pip install -r requirements.txt
    python ui.py

---

## How It Works

Neural networks see images as grids of numbers. By calculating the gradient of a model's prediction with respect to each pixel, Tangle identifies exactly which pixels to nudge to maximally confuse the model. The changes are mathematically significant but visually invisible.

---

## Stack

Python, PyTorch, OpenAI CLIP, CustomTkinter, Pillow

---

## Author

Built by redd2213.
