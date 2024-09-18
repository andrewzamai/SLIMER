<div align="center">
  <h1>👻 SLIMER: Show Less Instruct More Entity Recognition</h1>
</div>


<p align="center">
    <a href="https://github.com/andrewzamai/SLIMER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-Apache2.0-blue"></a>
    <a href="https://huggingface.co/expertai/SLIMER"><img alt="Models" src="https://img.shields.io/badge/🤗-Models-green"></a>
    <a href="https://arxiv.org/abs/2407.01272"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://www.expert.ai/"><img src="https://img.shields.io/badge/company-expert.ai-blueviolet"></a>
</p>

# Enrich Prompts with Definitions and Guidelines for Zero-Shot NER

Works on:

  ✅ Out-Of-Domain inputs

  ✅ Never-Seen-Before Named Entities 


<!-- Simplified container layout -->
<div align="center">
  <h2>Instruction Tuning Prompt</h2>
</div>

<!-- Content section -->
<p><b>[INST]</b></p>
<p>You are given a text chunk (delimited by triple quotes) and an instruction.<br>
Read the text and answer to the instruction in the end.</p>

<pre>
"""
{input text}
"""
</pre>

<p><b>Instruction:</b> Extract the Named Entities of type <b>DATE</b> from the text chunk you have read.</p>
<p>You are given a <b>DEFINITION</b> and some <b>GUIDELINES</b>.</p>

<!-- Definition box -->
<div style="background-color: #ccffcc; padding: 10px; border: 1px solid #000;">
  <p><b>DEFINITION:</b> <b>DATE</b> refers to specific points in time, including days, months, years, and relative time expressions like 'Week 2'.</p>
</div>

<!-- Guidelines box -->
<div style="background-color: #ffffcc; padding: 10px; border: 1px solid #000;">
  <p><b>GUIDELINES:</b> Avoid labeling non-specific time references like 'recently' or 'soon'. Exercise caution with ambiguous terms like 'May' (month or verb) and 'Wednesday Adams' (person's name which includes a day of the week).</p>
</div>

<p>Return a JSON list of instances of this Named Entity type. Return an empty list if no instances are present.</p>
<p><b>[/INST]</b></p>



## 📄 Abstract
Recently, several specialized instruction-tuned Large Language Models (LLMs) for Named Entity Recognition (NER) have emerged. Compared to traditional NER approaches, these models have demonstrated strong generalization capabilities. Existing LLMs primarily focus on addressing zero-shot NER on Out-of-Domain inputs, while fine-tuning on an extensive number of entity classes that often highly or completely overlap with test sets. In this work instead, we propose SLIMER, an approach designed to tackle never-seen-before entity tags by instructing the model on fewer examples, and by leveraging a prompt enriched with definition} and guidelines.
Experiments demonstrate that definition and guidelines yield better performance, faster and more robust learning, particularly when labelling unseen named entities. Furthermore, SLIMER performs comparably to state-of-the-art approaches in out-of-domain zero-shot NER, while being trained in a more fair, though certainly more challenging, setting.


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
