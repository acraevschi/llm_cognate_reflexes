# Running the reconstruction agent

This guide runs the complete agent harness against an LM Studio model on the
same machine. It covers a small custom dataset and a local Lexibank checkout.

## 1. Install the harness

From the repository root:

```bash
conda activate llm_reconstruction
pip install -e '.[agent]'
```

This installs the `cognate-reconstruct` command and the optional LiteLLM
provider adapter. No hosted API is required for LM Studio.

## 2. Start LM Studio

Choose a model that follows multi-turn tool calls reliably. Native tool-use
support is preferable; small instruction models may emit prose instead of the
required function calls.

Start the local server from LM Studio's Developer tab, or with the documented
[`lms server start`](https://lmstudio.ai/docs/cli/serve/server-start) command:

```bash
lms server start --port 1234
```

The OpenAI-compatible base URL is normally `http://localhost:1234/v1`. Check
the model IDs currently exposed by the server:

```bash
cognate-reconstruct lm-studio-models
```

The value printed here—not necessarily the display name in the application—is
the value to pass to `--model`. If LM Studio API authentication is enabled, add
`--api-key ...` to both model discovery and inference commands.

LM Studio exposes tool requests through its
[OpenAI-compatible API](https://lmstudio.ai/docs/developer/openai-compat) and
documents function calls through
[`/v1/chat/completions`](https://lmstudio.ai/docs/developer/openai-compat/tools).
The harness executes those requests locally against
the typed deterministic tools and returns their results to the model; the model
does not execute Python or shell commands.

## 3. Run the included custom-data example

The example input is [reconstruction_input.json](../examples/reconstruction_input.json).
Run it with a model ID returned by the previous command:

```bash
mkdir -p runs

cognate-reconstruct infer \
  --lm-studio \
  --model '<LM_STUDIO_MODEL_ID>' \
  --input examples/reconstruction_input.json \
  --output runs/reconstruction_result.json \
  --trajectories runs/trajectories.jsonl
```

Verbose mode is the default. The terminal trace looks like:

```text
[agent:PROTO] starting reconstruction
[agent:PROTO] requesting model turn 1
[agent:PROTO] model returned 1 tool call(s)
[agent:PROTO] calling tool list_concepts
[agent:PROTO] tool list_concepts succeeded
[agent:PROTO] calling tool get_alignments
[agent:PROTO] calling tool test_sound_law
[agent:PROTO] accepted reconstruction commit
[agent:PROTO] reconstructed 2 concepts
```

Tool arguments and structured results are printed beneath each action. Large
results are truncated after 4,000 characters in the terminal only; the complete
interaction remains in the trajectory JSONL. Use `--max-event-chars 20000` for
a longer terminal rendering, or `--quiet` to disable live events.

The command writes:

- `reconstruction_result.json`: full beams, best vocabulary for every internal
  node, reconstruction steps, and trajectories;
- `trajectories.jsonl`: one append-only, training-ready trajectory per completed
  internal node.

Each new run appends to the trajectory file. Use a new path when runs should be
kept separate.

## 4. Custom dataset format

The input is strict JSON corresponding to `WorkbenchPayload`:

```json
{
  "lexicons": [
    {
      "variety_id": "language_a",
      "name": "Language A",
      "forms": [
        {
          "form_id": "language_a:water",
          "variety_id": "language_a",
          "concept_id": "water",
          "segments": ["p", "a"],
          "cognate_set_id": "water-1"
        }
      ]
    },
    {
      "variety_id": "language_b",
      "name": "Language B",
      "forms": [
        {
          "form_id": "language_b:water",
          "variety_id": "language_b",
          "concept_id": "water",
          "segments": ["f", "a"],
          "cognate_set_id": "water-1"
        }
      ]
    }
  ],
  "concepts": [
    {"concept_id": "water", "gloss": "water"}
  ],
  "newick": "(language_a,language_b)PROTO;"
}
```

Requirements:

- `segments` must already contain phonetic tokens. The loader never guesses a
  segmentation by splitting raw orthography.
- Every form's `variety_id` must match its containing lexicon.
- Newick leaf labels must exactly match lexicon `variety_id` values.
- Internal nodes should be named when stable output IDs are wanted.
- Cognate-set IDs are optional, but known IDs prevent unrelated same-concept
  forms from being aligned together.
- Concept glosses and aliases are optional but make semantic search useful.
- Anchors are not required and the command defaults to advisory anchor policy.

If `newick` is omitted or `null`, the workbench induces a tree from lexical
distances. For research inference, an independently justified classification
tree is usually preferable.

## 5. Prepare a local Lexibank dataset

The CLI reads an existing Lexibank/CLDF checkout. It does not download or run a
dataset's `lexibank makecldf` process.

First inspect dataset-scoped variety IDs:

```bash
cognate-reconstruct list-lexibank-varieties \
  --dataset data/lexibank/iecor
```

The tab-separated output is:

```text
VARIETY_ID    LANGUAGE_NAME    FORM_COUNT    TREE_GLOTTOCODE
```

Prepare all usable varieties and let the workbench induce a tree:

```bash
cognate-reconstruct prepare-lexibank \
  --dataset data/lexibank/iecor \
  --output runs/iecor_input.json
```

For an initial local-model run, selecting a small coherent subset is strongly
recommended:

```bash
cognate-reconstruct prepare-lexibank \
  --dataset data/lexibank/iecor \
  --variety-id 'iecor:<ID_1>' \
  --variety-id 'iecor:<ID_2>' \
  --variety-id 'iecor:<ID_3>' \
  --output runs/iecor_subset.json
```

Then run `cognate-reconstruct infer` exactly as for custom JSON.

To use a supplied tree:

```bash
cognate-reconstruct prepare-lexibank \
  --dataset data/lexibank/iecor \
  --newick-file path/to/tree.nwk \
  --output runs/iecor_input.json
```

The supplied Newick currently must use the dataset-scoped `VARIETY_ID` values
printed by `list-lexibank-varieties`. Source Glottocodes are retained as
metadata but are not safe unique leaf IDs because historical and modern stages
can share one Glottocode.

Dataset-scoped IDs normally contain `:` and therefore must be quoted in Newick:

```text
('iecor:<ID_1>','iecor:<ID_2>','iecor:<ID_3>')PROTO;
```

The Lexibank adapter requires CLDF cognate assignments and existing `Segments`
or `Phonemic_Segments`. Datasets containing only raw forms or no cognate table
are rejected rather than silently converted into unreliable phonetic data.

## 6. Useful inference controls

```text
--beam-width 5              retained candidates per concept
--max-turns 24              model turns allowed at each internal node
--max-tool-calls 64         tool calls allowed at each internal node
--temperature 0.1           model sampling temperature
--timeout 300               provider request timeout in seconds
--anchor-policy advisory    ignore, advisory, or scored
--anchor-match-factor 100   used only with scored anchors
--quiet                     disable the default terminal trace
--no-preflight              skip the LM Studio /v1/models check
```

The tool loop is sequential and deterministic apart from model generation. A
larger family can make many model calls because every internal node gets its own
hypothesis-management session.

## 7. Troubleshooting

### The model returns prose and never calls a tool

Use a model with reliable tool/function calling, lower the temperature, and
inspect LM Studio's server logs with:

```bash
lms log stream
```

The harness will request another tool call, but
eventually stops at `--max-turns` instead of looping forever.

### A tool call repeatedly fails validation

Verbose output includes the exact Pydantic or sound-rule parser error returned
to the model. The model must correct the arguments; the harness does not weaken
schemas to accept malformed rules.

### The LM Studio model is not found

Run `cognate-reconstruct lm-studio-models`. Load or expose the intended model in
LM Studio, or use the exact ID that the endpoint reports. `--no-preflight` is
available for nonstandard LM Studio configurations.

### A Lexibank tree fails leaf validation

Use the first column from `list-lexibank-varieties` as Newick leaf labels. An
ordinary Glottocode-only tree cannot always be mapped automatically when the
dataset contains historical stages or multiple varieties sharing a Glottocode.

## Not implemented yet

- automatic Lexibank download/build orchestration;
- automatic, ambiguity-safe remapping of Glottolog leaf IDs to dataset-scoped
  Lexibank variety IDs;
- resume/checkpoint of a partially completed family after a provider failure;
- an anchor-file option in the CLI (anchors remain available in the Python API);
- automatic benchmarking of whether a local model is competent at the required
  multi-turn tool protocol;
- Hugging Face/TRL/Unsloth optimization. The emitted trajectory contract is
  intended to support that later work.
