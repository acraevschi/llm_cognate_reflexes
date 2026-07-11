# Lexibank Dataset Exceptions & Manual Overrides

This document tracks manual overrides, dataset-specific exceptions, and tree-splicing fixes introduced to resolve classification, glottocode, and cognate-linking errors.

---

## 1. Glottocode Mapping Overrides

To prevent classifier failures and incorrect family tree assignments, the following glottocode overrides are dynamically applied during dataset loading in [loader.py](file:///Users/acraev/Work/llm_cognate_reflexes/cognate_reflexes/data/loader.py):

### `tlopo`
* **Exception:** The dataset uses local IDs `"pan"` (for Proto-Austronesian) and `"poc"` (for Proto-Oceanic). The pipeline default fell back to matching these directly as Glottolog IDs, mapping them to Punjabi (`pan`, Indo-European) and Poqomam (`poc`, Mayan). This caused the dataset family to resolve to Indo-European, discarding all actual Austronesian languages.
* **Override:** 
  * `"pan"` &rarr; `"aust1307"` (Proto-Austronesian)
  * `"poc"` &rarr; `"ocea1241"` (Proto-Oceanic)

### `tuled`
* **Exception:** 13 Tupian languages were missing standard glottocodes in the source metadata, which excluded them from being placed in the Tupi classification tree.
* **Override:** Mapped the following local names to their closest valid Glottolog languoid codes:
  * `"Tenharim"` &rarr; `"tenh1241"`
  * `"Wirafed"` &rarr; `"wira1264"`
  * `"OldGuarani"` &rarr; `"oldg1234"`
  * `"Kampe"` &rarr; `"camp1260"`
  * `"Ramarama"` &rarr; `"itog1239"`
  * `"Apapokuva"` &rarr; `"apap1239"`
  * `"Piripkura"` &rarr; `"piri1253"`
  * `"Kawahiva"` &rarr; `"kawa1283"`
  * `"Karipuna"` &rarr; `"kari1312"`
  * `"MaweNatterer"` &rarr; `"sate1243"`
  * `"MundurukuNatterer"` &rarr; `"mund1330"`
  * `"ApiakaNatterer"` &rarr; `"apia1248"`
  * `"Arawine"` &rarr; `"araw1282"`

---

## 2. Cognate Linking / Alignment

### `sidwellvietic`
* **Exception:** The compiler of this dataset assigned reconstructed Proto-Vietic (`viet1250`) forms to unique, isolated cognate set IDs (e.g., cognate ID `1` for the proto-form of "I", and cognate ID `2` for all modern descendants). This led to a 0-cognate overlap in the triplet pipeline.
* **Override:** For each concept (`concepticon_id`), we automatically identify the cognate set containing the proto-form and merge/map it to the most common/frequent attested cognate set among descendant languages. This maps the reconstructed proto-forms to the correct attested cognate sets and yields **186** valid Vietic triplets.

---

## 3. Structural Tree Resolution Fixes

### Recursive Polytomy Resolver (`newick_utils.py`)
* **Exception:** The baseline top-down/zip-based `resolve_all_polytomies` implementation in [newick_utils.py](file:///Users/acraev/Work/llm_cognate_reflexes/cognate_reflexes/tree/newick_utils.py) had a design bug: when resolving an ancestor polytomy (e.g. `aust1305` root), it replaced the entire subtree with copies from a pre-generated Cartesian product, which orphaned downstream polytomies (like `viet1250`).
* **Fix:** Rewrote `resolve_all_polytomies` using a recursive bottom-up approach. It resolves child polytomies first, combines their candidate topologies, and then resolves the parent node. This correctly preserves descendant resolutions and allows nested proto-languages to be reconstructed.

---

## 4. Historical Reconstruction Targets

### `iecor` and future datasets with attested historical varieties
* **Issue:** A historical variety is normally a leaf in Glottolog's
  classification, so it cannot safely be converted into a proto-language or
  inferred to be an ancestor from classification alone. IE-CoR additionally
  reuses several Glottocodes for different historical and modern stages.
* **Policy:** Forms are keyed internally by a unique
  `"<dataset>:<LanguageTable.ID>"` variety ID. Glottocodes are retained only
  for tree lookup and provenance. Historical examples are generated solely
  from `data/historical_lineages.csv`, whose rows assign each target's known
  descendants to a *first-diverging child branch*. The manifest can also
  explicitly nominate a historical target when a dataset does not expose a
  standard `historical` column; name-based guesses never generate examples.
* **Validation:** A historical target is emitted only when at least two
  distinct descendant branches have usable forms and a shared cognate set.
  This rejects one-lineage chains such as Old Polish → Middle Polish → modern
  Polish. When both target and descendant have dates, their temporal order is
  also checked.
* **Automatic source trees:** A standard CLDF `TreeTable`/`MediaTable`, or a
  canonical `cldf/tree.nwk` file, is treated as an authoritative temporal
  tree. An internal non-root node whose resolved variety name is historical
  (for example, `Old X`) is converted into branch relations automatically.
  Extra tree locations can be registered in `data/temporal_trees.csv`.
* **Segmentation:** The loader prefers CLDF `Segments`, then a declared
  `Phonemic_Segments` fallback. It never splits raw `Form` values generically,
  because they can be orthographic. Segment provenance is retained in the raw
  JSON output.
