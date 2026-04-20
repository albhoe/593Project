### Notebook Structure

#### 1. **Setup & Data Loading**
   - Installs required dependencies
   - Mounts Google Drive to access AO3 dataset
   - Loads JSONL format data from: `/content/drive/MyDrive/593project/ao3_14400001-14500000.jsonl`
   - Preprocesses data by flattening metadata and removing unnecessary columns

#### 2. **Model Configuration**
   - Loads BART tokenizer and model for sequence classification
   - Configures LoRA with the following parameters:
     - `r=8` - LoRA attention dimension
     - `lora_alpha=16` - Scaling parameter
     - `target_modules=["q_proj", "v_proj"]` - Targets BART attention layers
     - `lora_dropout=0.1` - Dropout for regularization
   - Converts pandas DataFrame to Hugging Face Dataset
   - Tokenizes inputs with max length of 1024 tokens

#### 3. **Training**
   - Training configuration:
     - **Epochs**: 3
     - **Batch Size**: 1 (per device) with gradient accumulation of 4 steps
     - **Learning Rate**: 5e-5
     - **Optimizer**: AdamW with custom scheduler
     - **Precision**: bfloat16 for TPU compatibility
   - Uses Hugging Face `Trainer` API for streamlined training loop

#### 4. **Model Saving**
   - Saves fine-tuned PEFT model to: `/content/drive/MyDrive/593project/fine_tuned_bart_lora_classification_saved`
   - Saves tokenizer alongside the model for future inference

### How to Run

1. **In Google Colab**:
   - Click the "Open in Colab" badge in the notebook
   - Authenticate Google Drive access when prompted
   - Ensure dataset file exists at the specified path
   - Run cells sequentially

2. **Data Preparation**:
   - Dataset should be in JSONL format with text and metadata columns
   - Update the file path in the notebook to match your data location

3. **Monitor Training**:
   - Training logs are saved to `./logs_classification/`
   - Results are saved to `./results_classification/`

### Technical Notes
- The notebook uses **bfloat16** precision instead of fp16 for TPU compatibility
- Custom optimizer and learning rate scheduler are defined to avoid XLA-related errors
- LoRA significantly reduces trainable parameters compared to full fine-tuning
- The model is saved to CPU before persisting to Google Drive

### Output
- Fine-tuned BART model with LoRA adapters
- Tokenizer configuration
- Training logs and evaluation metrics

---

For questions or issues, please refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/) for transformers and [PEFT documentation](https://huggingface.co/docs/peft/) for LoRA implementation details.
