<div align="center">
  <h1>👻 SLIMER: Show Less Instruct More Entity Recognition</h1>
</div>


<p align="center">
    <a href="https://github.com/andrewzamai/SLIMER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-Apache2.0-blue"></a>
    <a href="https://huggingface.co/expertai/SLIMER"><img alt="Models" src="https://img.shields.io/badge/🤗-Models-green"></a>
    <a href="https://arxiv.org/abs/2407.01272"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://www.expert.ai/"><img src="https://img.shields.io/badge/company-expert.ai-blueviolet"></a>
</p>

## Instruct your LLM with Definitions and Guidelines for Zero-Shot NER 🔎 📖

Designed to work on:

&nbsp;&nbsp;&nbsp;&nbsp;✅ Out-Of-Domain inputs (e.g. news, science, politics, music ...)

&nbsp;&nbsp;&nbsp;&nbsp;✅ Never-Seen-Before Named Entities (the model was not trained on that tag? It will tag it anyway!)

<div align="center">
<img src="assets/SLIMER_prompt.png" alt="Alt text" style="max-width: 100%; width: 275px;">
</div>


## 📄 TL;DR
Existing models primarily focus on addressing zero-shot NER on Out-of-Domain inputs, while fine-tuning on an extensive number of entity classes that often highly or completely overlap with test sets in literature. 

But what happens in every-day uses when they encounter entities that were not included in training? As demonstrated on BUSTER, they are likely to under-perform.

In this work instead, we propose SLIMER, an approach designed to tackle never-seen-before entity tags by instructing the model on fewer examples, and by leveraging a prompt enriched with definition and guidelines.
Experiments demonstrate that definition and guidelines yield better performance, faster and more robust learning, particularly when labelling unseen named entities. 
Furthermore, SLIMER performs comparably to state-of-the-art approaches in out-of-domain zero-shot NER, while being trained in a more fair, though certainly more challenging, setting.


## 📚 Citation

If you find SLIMER useful in your work or research, please consider citing our paper:

```bibtex
@misc{zamai2024lessinstructmoreenriching,
      title={Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER}, 
      author={Andrew Zamai and Andrea Zugarini and Leonardo Rigutini and Marco Ernandes and Marco Maggini},
      year={2024},
      eprint={2407.01272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01272}, 
}
```
