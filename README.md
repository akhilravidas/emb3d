# emb3d

`emb3d` is a command-line utility that allows users to generate embeddings using models from OpenAI, Cohere and HuggingFace.

## Installation

```sh
pip install --upgrade emb3d
```

## Usage

```sh
 Usage: emb3d [OPTIONS] INPUT_FILE COMMAND [ARGS]...

 Generate embeddings for fun and profit.

╭─ Arguments ─────────────────────────────────────────────────────────────────────╮
│ *    input_file      PATH  Path to the input file. [default: None] [required]   │
╰─────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ --model-id                         TEXT     ID of the embedding model. Default  │
│                                             is 'text-embedding-ada-002'.        │
│                                             [default: None]                     │
│ --output_file              -o      PATH     Path to the output file. If not     │
│                                             provided, a default path will be    │
│                                             suggested.                          │
│                                             [default: None]                     │
│ --api-key                          TEXT     API key for the backend. If not     │
│                                             provided, it will be prompted or    │
│                                             fetched from environment variables. │
│                                             [default: None]                     │
│ --remote                                    Run the job remotely. Default is    │
│                                             local execution unless the model is │
│                                             one of OpenAI / Cohere's models.    │
│ --max-concurrent-requests          INTEGER  (Remote Execution) Maximum number   │
│                                             of concurrent requests for the      │
│                                             embedding task. Default is 1000.    │
│                                             [default: 1000]                     │
│ --help                                      Show this message and exit.         │
╰─────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────╮
│ config         Get or set a configuration value.                                │
╰─────────────────────────────────────────────────────────────────────────────────╯

```

```bash
emb3d [OPTIONS]
