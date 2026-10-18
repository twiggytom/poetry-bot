In-progress project to use LoRA fine-tuning to make a base model specialize in writing contemporary poetry.

Data was scraped from the Adroit Poetry journal, and thus is not published with this repo. Config files are specific to training LoRA adapters for 3 specific open-source base models without quantization (Qwen2.5-1.5b, Qwen2.5-8b, and Ministral3-8b). The models were chosen for their relatively small size due to the data and compute limitations, though I am looking to move into heavier models (potentially requiring quantization).

src/ includes the training script and inference script. --config flag for the training script specifies the model config you want to load in for colab, and --model specifies the specific model and it's LoRA adapters you want to converse with. config/ contains the training and inference settings for the model specified in the name.

Future work could potentially expand into creating an eval harness to evaluate the performance of different models, different config settings, etc.
