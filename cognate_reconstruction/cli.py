"""Command-line harness for local and hosted agentic reconstruction inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from cognate_reconstruction.agent import (
    AgenticNodeReconstructor,
    AgentOrchestrator,
    ConsoleEventSink,
    JsonlTrajectorySink,
    LiteLLMProvider,
    ReconstructionService,
)
from cognate_reconstruction.ingestion import (
    adapt_concept_metadata,
    adapt_dataset_forms,
    ingest_payload,
)
from cognate_reconstruction.schemas.ingestion import WorkbenchPayload
from cognate_reconstruction.schemas.rules import AnchorPolicy
from cognate_reconstruction.traversal import RuleBasedReconstructor
from cognate_reflexes.data.loader import CLDFLoader, DatasetForms

DEFAULT_LM_STUDIO_BASE = "http://localhost:1234/v1"


def _api_base(value: str) -> str:
    return value.rstrip("/")


def _lm_studio_models(api_base: str, api_key: str | None) -> tuple[str, ...]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{_api_base(api_base)}/models", headers=headers)
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - explicit local/user URL
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not query LM Studio at {_api_base(api_base)!r}: {error}"
        ) from error
    models = payload.get("data", []) if isinstance(payload, dict) else []
    return tuple(
        str(model["id"])
        for model in models
        if isinstance(model, dict) and model.get("id")
    )


def _load_lexibank(dataset_path: str) -> DatasetForms:
    path = Path(dataset_path).expanduser().resolve()
    loaded = CLDFLoader(path.parent).load_dataset(path)
    if loaded is None:
        raise ValueError(
            f"{path} has no loadable CLDF cognate data with tokenized segments"
        )
    return loaded


def _write_json(path: str | Path, content: str) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content + "\n", encoding="utf-8")


def _command_models(args: argparse.Namespace) -> None:
    for model_id in _lm_studio_models(args.api_base, args.api_key):
        print(model_id)


def _command_list_lexibank(args: argparse.Namespace) -> None:
    dataset = _load_lexibank(args.dataset)
    for lexicon in adapt_dataset_forms(dataset):
        print(
            "\t".join(
                (
                    lexicon.variety_id,
                    lexicon.name,
                    str(len(lexicon.forms)),
                    lexicon.tree_glottocode or "",
                )
            )
        )


def _command_prepare_lexibank(args: argparse.Namespace) -> None:
    dataset = _load_lexibank(args.dataset)
    lexicons = adapt_dataset_forms(dataset)
    if args.variety_id:
        selected = set(args.variety_id)
        available = {lexicon.variety_id for lexicon in lexicons}
        if unknown := sorted(selected - available):
            raise ValueError(f"unknown Lexibank variety IDs: {unknown}")
        lexicons = tuple(
            lexicon for lexicon in lexicons if lexicon.variety_id in selected
        )
    if len(lexicons) < 2:
        raise ValueError("preparation requires at least two selected varieties")
    newick = (
        Path(args.newick_file).expanduser().read_text(encoding="utf-8").strip()
        if args.newick_file
        else None
    )
    payload = WorkbenchPayload(
        lexicons=lexicons,
        concepts=adapt_concept_metadata(dataset),
        newick=newick,
    )
    _write_json(args.output, payload.model_dump_json(indent=2))
    tree_message = (
        f"using {args.newick_file}"
        if args.newick_file
        else "tree will be induced from lexical distances during inference"
    )
    print(
        f"wrote {args.output}: {len(lexicons)} varieties, "
        f"{sum(len(item.forms) for item in lexicons)} forms; {tree_message}",
        file=sys.stderr,
    )


def _command_infer(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser()
    payload = WorkbenchPayload.model_validate_json(input_path.read_text(encoding="utf-8"))
    dataset = ingest_payload(payload)

    completion_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "timeout": args.timeout,
    }
    model = args.model
    if args.lm_studio:
        api_base = _api_base(args.api_base or DEFAULT_LM_STUDIO_BASE)
        raw_model = model.removeprefix("openai/")
        if not args.no_preflight:
            available = _lm_studio_models(api_base, args.api_key)
            if raw_model not in available:
                rendered = ", ".join(available) if available else "no models reported"
                raise ValueError(
                    f"model {raw_model!r} is not reported by LM Studio; available: {rendered}"
                )
        model = f"openai/{raw_model}"
        completion_kwargs.update(
            {
                "api_base": api_base,
                # LiteLLM's OpenAI-compatible client expects a value even when
                # LM Studio authentication is disabled.
                "api_key": args.api_key or "lm-studio",
            }
        )
    else:
        if args.api_base:
            completion_kwargs["api_base"] = _api_base(args.api_base)
        if args.api_key:
            completion_kwargs["api_key"] = args.api_key

    provider = LiteLLMProvider(model, completion_kwargs=completion_kwargs)
    event_sink = (
        None
        if args.quiet
        else ConsoleEventSink(max_json_chars=args.max_event_chars)
    )
    orchestrator = AgentOrchestrator(
        provider,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        trajectory_sink=JsonlTrajectorySink(args.trajectories),
        event_sink=event_sink,
    )
    deterministic = RuleBasedReconstructor(
        beam_width=args.beam_width,
        anchor_policy=AnchorPolicy(args.anchor_policy),
        anchor_match_factor=args.anchor_match_factor,
    )
    service = ReconstructionService(
        AgenticNodeReconstructor(
            orchestrator,
            deterministic=deterministic,
        )
    )
    result = service.reconstruct_family(dataset)
    _write_json(args.output, result.model_dump_json(indent=2))
    print(
        f"wrote {args.output}: {len(result.internal_nodes)} reconstructed internal nodes",
        file=sys.stderr,
    )
    print(f"appended trajectories to {args.trajectories}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognate-reconstruct",
        description="Run the deterministic cognate-reconstruction agent harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser(
        "lm-studio-models",
        help="List model IDs exposed by an LM Studio local server.",
    )
    models.add_argument("--api-base", default=DEFAULT_LM_STUDIO_BASE)
    models.add_argument("--api-key")
    models.set_defaults(handler=_command_models)

    varieties = subparsers.add_parser(
        "list-lexibank-varieties",
        help="List dataset-scoped variety IDs in one local Lexibank checkout.",
    )
    varieties.add_argument("--dataset", required=True)
    varieties.set_defaults(handler=_command_list_lexibank)

    prepare = subparsers.add_parser(
        "prepare-lexibank",
        help="Convert one local Lexibank CLDF dataset to workbench JSON.",
    )
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--variety-id",
        action="append",
        help="Dataset-scoped variety ID to include; repeat to select a subset.",
    )
    prepare.add_argument(
        "--newick-file",
        help="Optional Newick whose leaf labels exactly match selected variety IDs.",
    )
    prepare.set_defaults(handler=_command_prepare_lexibank)

    infer = subparsers.add_parser(
        "infer",
        help="Reconstruct every internal node from workbench JSON.",
    )
    infer.add_argument("--input", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--output", default="reconstruction_result.json")
    infer.add_argument("--trajectories", default="trajectories.jsonl")
    infer.add_argument("--lm-studio", action="store_true")
    infer.add_argument("--api-base")
    infer.add_argument("--api-key")
    infer.add_argument("--no-preflight", action="store_true")
    infer.add_argument("--beam-width", type=int, default=5)
    infer.add_argument(
        "--anchor-policy",
        choices=[policy.value for policy in AnchorPolicy],
        default=AnchorPolicy.ADVISORY.value,
    )
    infer.add_argument("--anchor-match-factor", type=float, default=100.0)
    infer.add_argument("--temperature", type=float, default=0.1)
    infer.add_argument("--timeout", type=float, default=300.0)
    infer.add_argument("--max-turns", type=int, default=24)
    infer.add_argument("--max-tool-calls", type=int, default=64)
    infer.add_argument("--max-event-chars", type=int, default=4000)
    infer.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the default verbose model/tool trace.",
    )
    infer.set_defaults(handler=_command_infer)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
