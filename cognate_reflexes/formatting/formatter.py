"""Text formatter for converting TrainingExample objects to structured text.

Produces the human-readable (and model-friendly) representation that the
LLM sees during fine-tuning.  IPA segments are space-separated, forms are
delimited, and masked forms show a configurable placeholder token.

Example output (with glosses, no language names)::

    [INPUT_0] WATER: w ɔ t ə r, FIRE: f aɪ ə r, DOG: d ɒ ɡ
    [INPUT_1] WATER: a ɡ w a, FIRE: f w e ɡ o, DOG: p e r o
    [TARGET] WATER: ???, FIRE: f ɔ k u s, DOG: ???
"""

from __future__ import annotations

from dataclasses import dataclass

from cognate_reflexes.examples.models import Form, TrainingExample


@dataclass
class TextFormatter:
    """Configurable text formatter for training examples.

    Attributes:
        lang_prefix_base: Base prefix for input language lines. Will be formatted with index (e.g., [INPUT_{i}]).
        target_prefix: Prefix for the target language line.
        form_separator: Delimiter between forms within one language.
        gloss_separator: Delimiter between a concept gloss and its IPA.
        segment_separator: Delimiter between IPA segments within a form.
        line_separator: Delimiter between language lines.
        mask_token: Placeholder shown for masked target forms.
        include_glosses: Whether to prepend concept glosses to forms.
        show_language_names: Whether to append language names to prefixes.
    """

    # Delimiters
    lang_prefix_base: str = "[INPUT_{i}]"
    target_prefix: str = "[TARGET]"
    form_separator: str = ", "
    gloss_separator: str = ": "
    segment_separator: str = " "
    line_separator: str = "\n"

    # Masking
    mask_token: str = "???"

    # Options
    include_glosses: bool = True
    show_language_names: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_example(self, example: TrainingExample) -> tuple[str, str]:
        """Format an example into ``(input_text, target_text)``.

        ``input_text`` contains ``INPUT_0``, ``INPUT_1``, etc. and ``TARGET``
        with masked forms replaced by :attr:`mask_token`.

        ``target_text`` contains the ground truth for **masked** forms
        only (cognate reflex task) or **all** forms (reconstruction
        task).

        Args:
            example: The :class:`TrainingExample` to format.

        Returns:
            ``(input_text, target_text)`` ready for the fine-tuning
            dataset.
        """
        masked_set = set(example.masked_indices)

        # --- Input text ------------------------------------------------
        lines = []
        for i, input_data in enumerate(example.inputs):
            line = self._format_language(
                forms=input_data.forms,
                prefix=self.lang_prefix_base.format(i=i),
                language_name=input_data.name if self.show_language_names else None,
            )
            lines.append(line)
            
        line_target = self._format_language(
            forms=example.target.forms,
            prefix=self.target_prefix,
            masked_indices=masked_set,
            language_name=example.target.name if self.show_language_names else None,
        )
        lines.append(line_target)

        input_text = self.line_separator.join(lines)

        # --- Target text -----------------------------------------------
        target_text = self._format_target_ground_truth(example)

        return input_text, target_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_language(
        self,
        forms: list[Form],
        prefix: str,
        masked_indices: set[int] | None = None,
        language_name: str | None = None,
    ) -> str:
        """Format a single language's forms into one line."""
        masked_indices = masked_indices or set()

        # Build prefix with optional language name.
        header = prefix
        if language_name:
            # "[INPUT_A: French]" instead of "[INPUT_A]"
            header = f"{prefix[:-1]}: {language_name}]" if prefix.endswith("]") else f"{prefix} {language_name}"

        parts: list[str] = []
        for i, form in enumerate(forms):
            is_masked = i in masked_indices
            parts.append(self._format_form(form, masked=is_masked))

        return f"{header} {self.form_separator.join(parts)}"

    def _format_form(self, form: Form, masked: bool = False) -> str:
        """Format a single form.

        With glosses:  ``WATER: w ɔ t ə r``  or  ``WATER: ???``
        Without glosses:  ``w ɔ t ə r``  or  ``???``
        """
        if masked:
            ipa_str = self.mask_token
        else:
            ipa_str = self.segment_separator.join(form.segments)

        if self.include_glosses:
            return f"{form.concept}{self.gloss_separator}{ipa_str}"
        else:
            return ipa_str

    def _format_target_ground_truth(self, example: TrainingExample) -> str:
        """Format the ground truth answer.

        For cognate reflex: only the masked forms.
        For reconstruction: all target forms.
        """
        masked_set = set(example.masked_indices)
        parts: list[str] = []

        for i, form in enumerate(example.target.forms):
            if i in masked_set:
                ipa_str = self.segment_separator.join(form.segments)
                if self.include_glosses:
                    parts.append(
                        f"{form.concept}{self.gloss_separator}{ipa_str}"
                    )
                else:
                    parts.append(ipa_str)

        return self.form_separator.join(parts)

