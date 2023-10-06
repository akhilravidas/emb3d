# emb3d

`emb3d` is a command-line utility that allows users to generate embeddings using models from OpenAI, Cohere and HuggingFace.

## Installation

```sh
pip install --upgrade emb3d
```

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

```bash
emb3d [OPTIONS]
