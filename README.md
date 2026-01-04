---
license: llama2
base_model: NousResearch/Llama-2-7b-chat-hf
tags:
  - legal
  - constitution
  - nepal
  - lora
  - peft
  - fine-tuning
---

# LLaMA-2 Nepal Constitution (LoRA)

This repository contains **LoRA adapter weights** for fine-tuning LLaMA-2-7b on question-answer pairs derived from the **Constitution of Nepal**. The model has been optimized for legal education, constitutional reference, and research purposes.

## Overview

This project fine-tunes the LLaMA-2-7b-chat model using Low-Rank Adaptation (LoRA/PEFT) to specialize in answering questions about Nepal's Constitution. The training data consists of carefully curated question-answer pairs extracted from constitutional documents.

## Base Model

- **Model**: [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- **Parameters**: 7 billion
- **Framework**: Hugging Face Transformers + PEFT

## Dataset

- **Source**: Constitution of Nepal
- **Format**: JSONL (question-answer pairs)
- **Location**: `Data/QAPairs (3).jsonl`
- **Purpose**: Legal Q&A fine-tuning

## Project Structure

```
├── README.md                              # This file
├── 03_final_notebook.ipynb               # Main project notebook
├── Data/
│   └── QAPairs (3).jsonl                 # Training data (Q&A pairs)
├── Datapreparation/
│   └── 02_local_llm.ipynb                # Data preparation script
└── Kaggle/
    ├── 02-local-llm.ipynb                # Kaggle notebook version
    └── fine-turning-model.ipynb          # Fine-tuning implementation
```

## Intended Use

- **Legal Education**: Learning about Nepal's Constitutional framework
- **Constitutional Reference**: Quick access to constitutional information
- **Research & Experimentation**: Testing fine-tuning methodologies
- **Legal AI Applications**: Building domain-specific legal chatbots

## Model Capabilities

This fine-tuned model can:
- Answer questions about Nepal's Constitution
- Explain constitutional articles and provisions
- Provide context on constitutional amendments
- Assist in legal research and education

## Limitations

⚠️ **Important Disclaimers**:
- **Not a substitute for legal advice**: This model provides informational content only
- **Incomplete Information**: Answers may be incomplete and should be verified
- **Potential Errors**: May occasionally generate incorrect or hallucinated content
- **Domain Specific**: Optimized for Nepal's Constitution, performance may vary on other legal topics
- **Training Data**: Limited to provided Q&A pairs

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
bitsandbytes>=0.39.0
```

## Installation

```bash
# Clone or download this repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## How to Use

### Using the Fine-tuned LoRA Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
LORA_MODEL = "biswaa000/llama-2-nepal-constitution-lora"

# Load tokenizer (from LoRA repo)
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()

# Prompt
prompt = "What is the basic structure of Nepal's Constitution?"
formatted_prompt = f"<s>[INST] {prompt} [/INST]"

# Generate
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Clean echoed prompt
if "[/INST]" in response:
    response = response.split("[/INST]")[-1].strip()

print(response)

```

### Training Your Own LoRA Adapter

Refer to `Kaggle/fine-turning-model.ipynb` for the complete fine-tuning pipeline.

## Key Features

- ✅ **Efficient Fine-tuning**: Uses LoRA for parameter-efficient adaptation
- ✅ **Domain-Specific**: Trained on Nepal Constitution Q&A pairs
- ✅ **Reproducible**: Complete notebooks provided for data preparation and training
- ✅ **Easy Integration**: Drop-in replacement with standard transformers API

## Training Details

- **Method**: LoRA (Low-Rank Adaptation)
- **Adapter Configuration**: See notebooks for detailed hyperparameters
- **Data Format**: JSONL with question-answer pairs
- **Framework**: PyTorch + Hugging Face PEFT

## Notebooks

1. **`03_final_notebook.ipynb`**: Main project notebook with full pipeline
2. **`Datapreparation/02_local_llm.ipynb`**: Data preparation and preprocessing
3. **`Kaggle/02-local-llm.ipynb`**: Alternative implementation
4. **`Kaggle/fine-turning-model.ipynb`**: Fine-tuning code

## Performance

The model has been evaluated on:
- Constitutional knowledge accuracy
- Answer relevance and completeness
- Factual correctness against source documents

## Future Improvements

- [ ] Expand training data with more constitutional documents
- [ ] Fine-tune on related legal documents
- [ ] Implement RAG (Retrieval-Augmented Generation) for improved accuracy
- [ ] Support for other constitutional domains

## License

This project is provided under the **LLAMA-2 License**. See the base model's license for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{nepal_constitution_lora,
  title={LLaMA-2 Nepal Constitution LoRA Adapter},
  author={Your Name},
  year={2026},
  publisher={Hugging Face Model Hub}
}
```

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## Support & Contact

For questions or issues related to this project, please open an issue in the repository.

---

**Disclaimer**: This model is for educational and research purposes. Always verify critical legal information through official sources or consult with legal professionals.

**Last Updated**: January 2026
