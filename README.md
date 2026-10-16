In-progress project to use LoRA fine-tuning to make a base model specialize in writing contemporary poetry.

Data was scraped from the Adroit Poetry journal, and thus is not published with this repo. Base model is the Qwen2.5-1.5B model, since it's small enough to load without quantiazation, though I am looking to use slightly more powerful base models like the Llama 3B model or quantizing heavier models.

src/ includes the training script and inference script. config/ contains the training and inference settings.

Model is far from perfect. Future work will focus on scraping more data, training on a larger base model, and optimizing config settings.
