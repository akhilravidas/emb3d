# emb3d

`emb3d` is a command-line utility that lets you generate embeddings using models from OpenAI, Cohere and HuggingFace.

## Quick Start ⚡️

### Install the library

```sh
pip install --upgrade emb3d
```

### Prepare your input file

emb3d expects a JSONL file as input. Each line of the file should be a JSON object with a `text` key. Example input file:

```json
{"text": "I love my dog"}
{"text": "I love my cat"}
{"text": "I love my rabbit"}
```

Your files can optionally have other fields like ids, categorical labels etc.. and they are saved as-is in the final output file.

### Compute embeddings

The default model is OpenAI's `text-embedding-ada-002`. You can change the model by passing the `--model-id` flag.

```sh
emb3d compute inputs.jsonl
```

<img width="1050" alt="Xnapper-2023-10-06-16 15 47" src="https://github.com/akhilravidas/emb3d/assets/104069/a1939269-906f-4bf5-b10e-7771ec7d7556">

You will need to have OPENAI_API_KEY set in your environment. You can also pass it as a flag (`--api_key`) or set it in a config file.

```sh:
emb3d config set openai_token YOUR-OPENAI-API-KEY
emb3d compute inputs.jsonl
```

```sh
emb3d compute inputs.jsonl --model-id embed-english-v2.0 --output-file cohere-embeddings.jsonl
```

For COHERE models, you will need to have COHERE_API_KEY set in your environment. You can also pass it as a flag (`--api_key`) or set it in a config file with: `emb3d config set cohere_token YOUR-COHERE-API-KEY`.



### Visualize your embeddings 💥

The last step is to visualize your computed embeddings. This will open a browser window with a visualization of your last computed embeddings.

```sh
emb3d visualize run-2020-embeddings.jsonl
```

### Profit 💰

## Usage

```
 Usage: emb3d [OPTIONS] INPUT_FILE COMMAND [ARGS]...

 Generate embeddings for fun and profit.

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    input_file      PATH  Path to the input file. [default: None] [required]             │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --model-id                                     TEXT     ID of the embedding model.        │
│                                                         Default is                        │
│                                                         `text-embedding-ada-002`.         │
│                                                         [default: None]                   │
│ --output-file              -out,-o             PATH     Path to the output file. If not   │
│                                                         provided, a default path will be  │
│                                                         suggested.                        │
│                                                         [default: None]                   │
│ --api-key                                      TEXT     API key for the backend. If not   │
│                                                         provided, it will be prompted or  │
│                                                         fetched from environment          │
│                                                         variables.                        │
│                                                         [default: None]                   │
│ --remote                            --local             Choose whether to do inference    │
│                                                         locally or with an API token.     │
│                                                         This choice is available for      │
│                                                         sentence transformer and hugging  │
│                                                         face models. If a model cannot be │
│                                                         run locally (ex: OpenAI models),  │
│                                                         this flag is ignored.             │
│                                                         [default: remote]                 │
│ --max-concurrent-requests                      INTEGER  (Remote Execution) Maximum number │
│                                                         of concurrent requests for the    │
│                                                         embedding task. Default is 1000.  │
│                                                         [default: 1000]                   │
│ --help                                                  Show this message and exit.       │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────╮
│ config           Get or set a configuration value.                                        │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```
<img width="1050" alt="Xnapper-2023-10-06-15 30 13" src="https://github.com/akhilravidas/emb3d/assets/104069/41cd9b27-ff53-420f-bedf-a85c3d4c769d">




## Need help? 🙋

Join our [Discord server](https://discord.gg/qncFtMxP) and lets talk!

## Contributing

If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. You can also reach us at our [Discord server](https://discord.gg/qncFtMxP).

## License

emb3d CLI tool is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
