# emb3d

`emb3d` is a command-line utility that lets you generate embeddings using models from OpenAI, Cohere and HuggingFace.

## Installation

```sh
pip install --upgrade emb3d
```

## Quick Start ⚡️

### Install the library

```sh
pip install -U emb3d
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

The last step is to visualize your embeddings. This will open a browser window with a visualization of your last computed embeddings.
```sh
emb3d visualize
```

You can alternatively pass the path to the computed embeddings file:

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
