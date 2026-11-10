# Plan: Comparison Mode for the Gallery Panel

## 1. Goal

The app is used to evaluate outputs from different models. Today, finding "the same image,
but rendered by model B" requires switching galleries and scrolling, or typing the prompt
into search — both fiddly because several images can share a prompt.

This feature adds a **comparison mode** to the right-hand gallery panel. When an image is
displayed in the main view, the user can switch the gallery panel into comparison mode. The
panel then shows **all images, from any gallery, that match the displayed image on a
user-defined set of generation parameters**, restricted to a user-defined set of models.
Matching is defined by hashing a canonical JSON object built from the selected fields
(the "identity metric"): two images match iff their hashes are equal.

## 2. User-visible behavior (acceptance criteria)

1. A new toggle button (⇄, tooltip "Compare across models") appears in the gallery header.
   Clicking it switches the panel between **Browse mode** (today's behavior, unchanged) and
   **Compare mode**.
2. Comparison mode requires a *reference image*: the currently selected image in the main
   view (`selectedImageName`). If none is selected, fall back to the first loaded image of
   the current board. If there is no image at all, the toggle button is disabled.
3. In comparison mode the board dropdown, "new gallery" button and search box are hidden and
   replaced by a **config panel** with two collapsible checkbox groups:
   - **Group 1 — "Match by"** (collapsible, open by default): checkboxes for the fields
     `prompt`, `negative prompt`, `seed`, `width`, `height`, `CFG`, `model`.
     Defaults: all checked **except** `model`.
   - **Group 2 — "Models"** (collapsible, open by default): one checkbox per model known to
     the system (all registered models plus removed models that still have images; see §3.2),
     plus **Select all** / **Select none** buttons. Default selection: the **anchor model +
     the last two models added to the system** (so 2 or 3 checked, depending on whether the
     anchor is already among the last two). The selection persists across sessions and
     re-entries; newly added models join it pre-checked. Precise rules in §3.2.
4. Below the config panel, the thumbnail grid lists **every image matching the reference
   image's hash** (computed from the checked fields) whose `model_key` is among the checked
   models, across **all galleries**. The grid re-queries automatically whenever any checkbox
   changes. Order: newest submission first (`request_serial_number` desc), same as the
   default browse sort. The count row shows `N images`.
5. Each thumbnail carries a small badge with the generating model's name so matches from
   different models are distinguishable at a glance. The reference image is **always** in
   the results (its model is anchor-checked on entry, §3.2) and highlighted like a normal
   selection.
6. All existing gallery interactions keep working in comparison mode: click to display in
   the main view, shift/cmd multi-select, star/delete hover buttons, right-click context
   menu, infinite scroll, jump navigation (⏮ −500 … ⏭), arrow-key navigation.
7. Exiting comparison mode restores browse mode on the previously selected board.
8. Newly generated images arriving over WebSocket while in comparison mode trigger a
   debounced reload of the comparison results.

## 3. Design decisions (with rationale)

### 3.1 Hashing is computed at query time, in Python, not stored

The identity metric is user-configurable (any subset of 7 fields ⇒ 128 combinations), so a
single stored hash column cannot serve it. Options considered:

- **Store one hash per field-subset combination** — 128 columns or a side table; absurd
  complexity for a single-user tool.
- **SQL equality (`WHERE prompt = :p AND seed = :s …`)** — equivalent results, but it does
  not give us the hash the user explicitly specified, and it silently diverges from hashing
  on NULL-vs-empty-string edge cases.
- **Chosen: compute the hash in Python at query time.** Load the candidate rows (already
  narrowed by `model_key IN (...)`), build the canonical JSON from the checked fields, take
  SHA-256 hex, and keep rows whose hash equals the reference's. Data volumes are a single
  user's generation history (thousands of rows at most), so the scan is negligible, and the
  endpoint stays honest to the spec.

### 3.2 Model selection = anchor + last two added, persisted

**User-clarified semantics:** the checked set is **{anchor image's model} ∪ {the last two
models added to the system}** (2 or 3 models; "added to the system", not "recently used");
afterwards the selection **persists**, and newly added models are checked automatically.

The only "added" signal in the system is **insertion order in `models.json`**:
`ModelRegistry` stores entries in a plain dict keyed by UUID, `all()` returns
`list(self._entries.values())`, and the JSON round-trip preserves order (Python dicts and
Python's JSON reader are insertion-ordered). There is **no `added_at` field on
`ModelEntry`** — do not invent one; adding a timestamp column to the registry is out of
scope and would not help for models registered before this feature. So "last two added" :=
**the final two elements of `registry.all()`** after filtering to real models.

Candidate list shown in Group 2 ("Models") is the union of two sources, so every model the
user can meaningfully compare against appears:

1. **Registered models** — `request.app.state.registry.all()` filtered to
   `entry.kind != "lora"` (includes merge entries; they render images). Listed in registry
   insertion order. Current `entry.name` is shown (rename-proof, unlike the DB snapshot).
   Their image count comes from the DB and may be **0** for a just-installed model — it is
   still listed, because comparing *to* a fresh model is the point of this feature.
2. **Removed models that still have images** — distinct `model_key`s in the `images` table
   not present in the registry (user deleted the registry entry). Appended **after** the
   registered list, ordered by most recent image (`MAX(created_at)` desc), labeled with
   their latest `model_name` snapshot and flagged `registered: false` so the UI can render
   them dimmed with a "(removed)" suffix. A removed model is never *added*, so it only
   ends up default-checked when it is the anchor model of the current reference image, or
   when it was checked while still registered and survives pruning.

Default / ongoing model selection (user-specified, two parts):

**A. First entry ever** (no persisted selection — see §5.1 for how "never persisted" is
distinguished from "persisted empty"):
`selection = {anchor model} ∪ {last two added}` (set union, so 2 or 3 checked depending on
whether the anchor is one of the last two). "Anchor model" = the `model_key` of the
reference image R. Edge cases:
- Fewer than two registered models: use whatever registered models exist (plus anchor).
- Zero registered models (all deleted): the "last two added" part is empty; the candidate
  list is the removed tail only, and selection = {anchor}.

**B. Every later entry** (a persisted selection exists, empty or not):
`selection = prune(persisted) ∪ {anchor model} ∪ {newly added models}` where
- `prune(persisted)` = checked keys that still appear in the candidate list (a key vanishes
  only if the model was deleted **and** all its images were deleted);
- `newly added models` = keys present in the fresh candidate list but absent from the
  persisted **known-keys snapshot** (see §5.1 `compareKnownModels`) — i.e. models installed
  since the last compare-mode entry. A registered model moving to the removed tail keeps
  its key, so deletion is not treated as "added".
Manual unchecks therefore survive re-entry (the model is known, not new, and stays pruned
in), and every freshly installed model is checked by default.

Consequence of the anchor term (intended): the reference image's model is always checked
when comparison mode is entered, so R itself always appears in the results grid — this was
the point of comparison in the first place — while persistence keeps the user's manual
list stable across sessions.

Known caveat, acceptable: if a user deletes a middle registry entry and re-adds a model, it
lands at the end of the order — "last added" then means last *re*added. That is the natural
reading of the registry state.

### 3.3 `model` as a match field maps to `model_key`, not `model_name`

`model_key` is the stable UUID per registry entry (`ImageRecord.model_key`); `model_name` is
a snapshot that changes if the user renames the model in the Models tab. Keying both the hash
and the model filter on `model_key` keeps them consistent. (If `model` is checked in "Match
by", results only ever contain R's own model — that is the documented, intended semantics of
the identity metric.)

### 3.4 Canonical JSON spec (must be implemented exactly — this defines identity)

- Fixed set of canonical field names, always emitted in this order-independent way: build a
  `dict` keyed by the canonical names, then serialize with
  `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
- Canonical names and column mapping:
  | checkbox id | canonical key | `ImageRecord` column |
  |---|---|---|
  | `prompt` | `prompt` | `prompt` |
  | `negative_prompt` | `negative_prompt` | `negative_prompt` |
  | `seed` | `seed` | `seed` |
  | `width` | `width` | `width` |
  | `height` | `height` | `height` |
  | `cfg` | `cfg` | `cfg` |
  | `model` | `model` | `model_key` |
- **NULL normalization**: `None` values normalize to `""` for string columns
  (`prompt`, `negative_prompt`, `model_key`) and to `0` for `seed`, `width`, `height`, and
  `0.0` for `cfg`. Rationale: legacy rows may have NULLs while newer rows have defaults;
  without normalization a NULL/"" pair would hash differently despite being the same image
  parameters. `steps`, `sampler`, LoRAs etc. are **not** part of the metric in this feature.
- Hash = `hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()`.
- The request `fields` list may specify any subset, including the empty set. Empty set ⇒
  every candidate row has the same hash ⇒ returns all images of the selected models
  (deliberate: "compare everything these models made"). This is allowed; the UI never
  deselects all fields silently, but the API accepts it.

### 3.5 Backend does a plain scan + in-memory paginate; ordering done in SQL first

The service method orders candidates by `request_serial_number DESC NULLS LAST` in SQL,
hash-filters in Python (order is preserved), then slices for `offset/limit` and returns
`(total, page_items)`. `total` is the number of hash-matches, which drives the existing
pagination/jump UI unchanged.

### 3.6 `navigate-board` exits comparison mode

The info-overlay "↗ Show in gallery" flow (`navigate-board` DOM event) is board-scoped by
nature. If it fires while in comparison mode, switch to browse mode first, then navigate.
This keeps that flow's semantics intact and avoids re-implementing page-scanning search
inside comparison results.

## 4. Backend work

### 4.1 New module: `backend/gallery/compare.py`

Single responsibility: canonical hashing. Pure functions, no DB access.

```python
FIELD_COLUMNS: dict[str, tuple[str, object]]  # canonical name -> (attr name on ImageRecord, null-normalization value)
ALL_FIELDS: list[str]  # ["cfg", "height", "model", "negative_prompt", "prompt", "seed", "width"]

def validation_error_fields(fields: list[str]) -> list[str]: ...   # unknown names; raise ValueError("unknown comparison fields: ...")
def canonical_payload(record_or_dict, fields: list[str]) -> str: ... # JSON string per §3.4
def comparison_hash(record, fields: list[str]) -> str: ...           # sha256 hexdigest of canonical_payload
```

- Read field values via `getattr(record, FIELD_COLUMNS[f][0])`.
- Raise `ValueError` on unknown field names (the API layer maps this to HTTP 400).

### 4.2 `backend/gallery/service.py` — two new `GalleryService` methods

```python
async def model_image_stats(self) -> dict[str, dict]:
    """Per-model aggregates from the images table (DB side of §3.2).

    SELECT model_key, COUNT(*) AS image_count, MAX(created_at) AS last_used
    FROM images WHERE model_key IS NOT NULL AND model_key != ''
    GROUP BY model_key

    model_name: resolve per model_key from the newest row in that group
    (name may change via registry rename; latest snapshot wins).

    Return: {model_key: {"model_name": str, "image_count": int,
                         "last_used": str}}  (time format like _image_record_to_dict)
    Skip img2img source-board rows: they have empty model_key.
    Ordering is not the service's concern — the route orders per §3.2/§4.3.
    """
```

```python
async def list_matching_images(
    self,
    reference: ImageRecord,
    fields: list[str],
    model_keys: list[str],
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[ImageRecord], int, str]:
    """Images across ALL boards whose hash of `fields` equals reference's hash.

    1. ref_hash = comparison_hash(reference, fields)
    2. query = select(ImageRecord).options(selectinload(tags))
       .where(ImageRecord.model_key.in_(model_keys))   # if model_keys empty → no filter
       .order_by(ImageRecord.request_serial_number.desc().nullslast())
    3. matches = [r for r in rows if comparison_hash(r, fields) == ref_hash]
    4. return matches[offset:offset+limit], len(matches), ref_hash
    """
```

Note: `selectinload(ImageRecord.tags)` keeps `_image_record_to_dict` from lazy-loading in an
async context (the pattern `list_images` already uses).

### 4.3 `backend/api/images.py` — two new endpoints

Both are `GET`, declared **before** the `/i/{image_name}` routes (cosmetic; no path
collision since both literals are top-level, but keep the file grouped: list-style endpoints
near `GET /`).

**`GET /api/v1/images/comparison_models`**

Assembles the §3.2 candidate list. Takes `request: Request` (for
`request.app.state.registry`) and `session=Depends(get_session)`:

1. `stats = await service.model_image_stats()`
2. Registered entries: `[e for e in registry.all() if e.kind != "lora"]` (registry
   insertion order = added order). For each:
   `{"model_key": e.key, "model_name": e.name, "registered": True,
     "image_count": stats[e.key]["image_count"] if present else 0,
     "last_used": ... or None}`.
3. Orphan tail: stats keys not in the registry, sorted `last_used` desc, same dict shape
   with `"registered": False` and `model_name` from the stats snapshot.
4. Return `{"models": registered_list + orphan_list}`.

Note: `GET /api/v2/models/` already lists the registry but has no image counts and no DB
orphans — that is why this dedicated endpoint exists.

**`GET /api/v1/images/compare`**

Query params (FastAPI `Query` for the list types):
| param | type | meaning |
|---|---|---|
| `reference` | `str` (required) | `image_name` of the displayed image |
| `fields` | `list[str]`, default `["prompt","negative_prompt","seed","width","height","cfg"]` | canonical names from §3.4 |
| `models` | `list[str]`, default `[]` | model_keys; empty ⇒ no model restriction |
| `offset` | `int`, default 0 | |
| `limit` | `int`, default 50 | same cap semantics as `GET /` |

Behavior:
- Unknown name in `fields` → HTTP 400 with detail listing offending names.
- `reference` not found → HTTP 404 (same message style as `/i/{name}/metadata`).
- Response mirrors `GET /api/v1/images/` so the frontend grid code is shared:

```json
{
  "total": 3,
  "offset": 0,
  "limit": 50,
  "items": [ ..._image_record_to_dict(r)... ],
  "comparison_hash": "e3b0c4...",
  "fields": ["prompt","seed","width","height","cfg","negative_prompt"]
}
```

Add `"board_id"` and `"board_name"` to each item **only for compare-endpoint items**: do this
in the route by loading the record's gallery (`get_image_with_gallery`-style selectinload)
— browse-mode responses must not change shape for existing callers. Simplest compliant
approach: have `list_matching_images` also eager-load `ImageRecord.gallery` and add a small
`_image_record_to_dict`-adjacent helper in `images.py` (e.g. `_comparison_item_to_dict`)
that extends the common dict with `board_id`/`board_name`. Do not duplicate the base
function.

### 4.4 `README_API.md`

Add the two endpoints (params + response shapes) to the API reference, in the images
section, matching existing entry style.

## 5. Frontend work

### 5.1 `frontend/gallery.js` — new state and methods on `gallery()`

New state (declared near the top with the other fields):

```js
mode: 'browse',                 // 'browse' | 'compare'
compareModelsPersisted: false,  // set by _restoreGalleryUiState; true iff a saved selection exists (§3.2-A/B switch)
compareRef: null,               // image_name of the reference image
compareFields: ['prompt','negative_prompt','seed','width','height','cfg'], // 'model' unchecked by default
compareModels: [],              // model_keys currently checked (persisted; §3.2 rules)
compareKnownModels: [],         // model_keys seen at last compare-mode entry (persisted;
                                //   its absence in the blob marks "first entry ever")
compareAllModels: [],           // [{model_key, model_name, image_count, last_used, registered}] from comparison_models
compareConfig: { matchOpen: true, modelsOpen: true },  // collapsible group state
compareLoading: false,
```

Constants: `COMPARE_FIELD_OPTIONS` — array of `{id, label}` for the 7 fields
(`prompt`, `negative_prompt` → "Negative prompt", `seed`, `width`, `height`,
`cfg` → "CFG", `model` → "Model").

New methods:

- **`toggleCompareMode()`** — if `mode === 'browse'`: resolve the reference image
  (`this.selectedImageName || this.images[0]?.image_name`); if none, return. Set
  `compareRef`, `mode = 'compare'`, always re-fetch `comparison_models` into
  `compareAllModels` (counts and names change as models are used/renamed/installed), then
  call `_applyCompareSelection(anchorKey)` (§3.2 rules), then `resetAndLoad()` and
  `_persistGalleryUiState()` so the updated `compareModels`/`compareKnownModels` land in
  storage even if no checkbox is touched. When switching back: `mode = 'browse'`,
  `resetAndLoad()` (browse fetch path unchanged). `anchorKey` = the `model_key` of the
  reference image record (available from the loaded `images` entry; if absent from the
  current page e.g. after `navigate-board`, fetch
  `/api/v1/images/i/{compareRef}/metadata` to get it).
- **`_applyCompareSelection(anchorKey)`** — implements §3.2.
  `firstEntry = !this.compareModelsPersisted` — a plain boolean component field, set to
  `true` by `_restoreGalleryUiState()` iff the saved blob contains a `compare:models`
  array (this distinguishes "user has a saved selection, possibly empty" from "never
  configured"; `compareModelsPersisted` is component state only, never written to storage).
  - Registered keys list: `compareAllModels.filter(m => m.registered).map(m => m.model_key)`
    in list order (= added order); `lastTwoAdded` = its final two entries, or fewer.
  - If `firstEntry`: `compareModels = dedupe([anchorKey, ...lastTwoAdded])`.
    Else: `allKeys = compareAllModels.map(m => m.model_key)`;
    `compareModels = dedupe([
        ...this.compareModels.filter(k => allKeys.includes(k)),   // prune persisted
        anchorKey,
        ...allKeys.filter(k => !this.compareKnownModels.includes(k))])`  // newly added
  - Then `compareKnownModels = allKeys` in both branches, and
    `compareModelsPersisted = true`.
- **`onCompareConfigChanged()`** — bound to every checkbox change; `resetAndLoad()`.
- **`compareSelectAllModels()` / `compareSelectNoneModels()`** — set `compareModels` to all /
  `[]`, then `onCompareConfigChanged()`.
- **`loadComparePage()`** — async; builds
  `/api/v1/images/compare?reference=…&fields=…&fields=…&models=…&models=…&offset=…&limit=…`
  (repeat the param name per value, FastAPI list style; `encodeURIComponent` not needed —
  URLSearchParams handles it), fetches, appends `data.items` to `this.images`, sets
  `this.total`, advances `this.offset`, sets `this.hasMore = batch.length === this.limit`.
  Same loading/finally bookkeeping as `loadMore()`.
- Existing-method changes:
  - **`loadMore()`**: change the guard `!this.selectedBoardId` to also allow compare mode;
    when `this.mode === 'compare'` delegate to `loadComparePage()` and return. Keep the
    `_pendingRestore` logic browse-mode-only (in compare mode skip the per-board state
    restore; instead after load, if `compareRef` is present in `this.images`, call
    `selectImage(compareRef, idx)` on first page).
  - **`_handleWsEvent()`**: in the `image_created` branch, when `mode === 'compare'`, do not
    `_insertImageInOrder` (the new image may not match the hash); instead call a debounced
    `this.resetAndLoad()` (~800 ms, reuse a timer field; a plain instance property, no
    requestAnimationFrame needed). `image_deleted` / `images_moved` filtering by
    `image_name` keeps working as-is against `this.images`.
  - **`navigate-board` listener**: set `this.mode = 'browse'` at the top of the handler
    (§3.6).
  - **`_persistGalleryUiState()` / `_restoreGalleryUiState()`**: persist `compareFields`,
    `compareModels`, `compareKnownModels` and `compareConfig` under new keys
    (`compare:fields`, `compare:models`, `compare:known`, `compare:groups`) in the same
    `GALLERY_STATE_KEY` blob. Do **not** persist `mode` — after reload there is no
    guaranteed reference image, so the panel starts in browse mode; an ignore-mode rule
    also means a persisted `'compare'` value from a malformed blob is harmless. Add
    `compareFields`, `compareModels`, `compareKnownModels` and `compareConfig` to the
    state object built by `_persistGalleryUiState()` and to the watcher-driven
    `_persistGalleryUiState()` trigger list so checkbox changes debounce-save them;
    `toggleCompareMode()` additionally calls it once at entry (see above). On restore
    (`_restoreGalleryUiState`): set `compareModelsPersisted = Array.isArray(saved['compare:models'])`;
    load `compareModels`/`compareKnownModels` only when they are arrays; ignore stored
    `compareFields` names not in `COMPARE_FIELD_OPTIONS` (a future field-set change must
    not poison state) and, if that leaves `compareFields` empty, restore the default set
    (all fields except `model`).
  - **`setupArrowNav()`**, `handleThumbClick`, multi-select, star, delete, regenerate modal,
    context menu: no changes needed; they all operate on `this.images` / names.
    (`confirmMove` in compare mode works because `move_images` ignores the source board
    except in the broadcast payload.)

### 5.2 `frontend/index.html` — gallery panel markup

- Add to the header row (after the ⚙ button):
  `<button class="icon-btn" title="Compare across models" :disabled="!selectedImageName && images.length === 0" :class="{active: mode==='compare'}" @click="toggleCompareMode()">⇄</button>`
- Wrap the existing board-dropdown row, the "+" button and the search row in
  `x-show="mode === 'browse'"` (hide the whole first two rows' contents that are
  browse-specific; keep the ⚙/collapse/⇄ buttons always visible by moving them into a
  small always-present row or giving the row `x-show` only around the dropdown).
- Add the **config panel** block `x-show="mode === 'compare'"` between the header and the
  selection bar:
  - Group 1: collapsible section. Header button toggles `compareConfig.matchOpen`
    (caret ▾/▸), label "Match by". Body: `x-for` over `COMPARE_FIELD_OPTIONS`, each a
    `<label class="checkbox-label"><input type="checkbox" :value="f.id"
    x-model="compareFields" @change="onCompareConfigChanged()"> label</label>`.
    (`x-model` on a checkbox array gives add/remove-by-value automatically.)
  - Group 2: collapsible header "Models", body contains Select all / Select none buttons
    and `x-for` over `compareAllModels`: checkbox bound to `compareModels` (same array
    binding), label `model_name`, count suffix `(image_count)`; entries with
    `registered === false` render dimmed with a " (removed)" suffix, entries with
    `image_count === 0` show `(no images yet)`. Body wrapped in a
    `max-height: 160px; overflow-y: auto` div so many models don't push the grid away.
  - A hint row under the config: `x-text="loading ? 'Loading…' : `${total} matching images`"`.
- Thumbnail grid: add the model badge overlay inside `thumb-wrapper`:
  `<span class="thumb-model-badge" x-show="mode === 'compare'" x-text="img.model_name"></span>`
  (truncated by CSS ellipsis; full name in a `:title`).
- The `gallery-nav-row` (jump buttons), `selection-bar`, `#gallery-contents`, sentinel and
  context menu stay exactly as-is — they are mode-agnostic.

### 5.3 `frontend/style.css` — new classes

Follow the existing design-system variables used by `#gallery-panel` components
(same tokens the `gallery-options-popup` / `checkbox-label` rows already use):

- `.compare-config` — bordered block below the header, padding, gap.
- `.compare-group-header` — clickable, caret + label, hover background.
- `.compare-group-body` — indented, padded.
- `.compare-models-scroll` — `max-height:160px; overflow-y:auto`.
- `.compare-btn-row` — small buttons (reuse `.icon-btn sm` / existing `.btn-secondary sm`
  for Select all/none if the design fits; otherwise minimal new class).
- `.thumb-model-badge` — absolute-positioned bottom strip inside `.thumb-wrapper`,
  small font, semi-transparent background, `text-overflow: ellipsis`, `white-space: nowrap`,
  `overflow: hidden`.
- `.icon-btn.active` (if not already present) — pressed-state styling for the ⇄ toggle.

## 6. Tests

### 6.1 Unit: `tests/unit/test_compare_hash.py` (new)

Using plain dicts or a lightweight stub object (not the ORM) since `compare.py` only does
`getattr`:

1. `test_hash_independent_of_dict_build_order` — two records with equal selected values
   hash equal even when the `fields` lists differ in order (hashing must sort keys).
2. `test_hash_differs_when_unchecked_field_differs_ignored` — rows differing only in
   `steps`/`sampler` (not in the metric) hash equal.
3. `test_hash_null_normalization` — `prompt=None` vs `prompt=""` hash equal; `seed=None` vs
   `seed=0` hash equal.
4. `test_hash_empty_fields` — `fields=[]` produces a constant hash for any two records.
5. `test_unknown_field_raises` — `ValueError` for `"steps"`.

### 6.2 Integration: `tests/integration/test_api_compare.py` (new, uses the `client` fixture)

Seed images directly via `GalleryService.record_image(...)` or raw ORM inserts through the
test DB (mirror how `tests/integration/test_api_images.py` seeds records). Register models
A and B through `POST /api/v2/models/install?source=…` (in that order, so B is the "last
added") — the `client` fixture gives a real `ModelRegistry` backed by `tmp_path/models.json`.
Fixture data: model A and model B each render prompt "P", seed 42, 512×512, cfg 7.5; model B
also renders prompt "P" with seed 43; one img2img-source-style row with empty `model_key`;
one row with `model_key="orphan"` that is NOT in the registry (simulates a deleted model
whose images remain).

Cases:
1. `test_compare_default_fields_matches_other_model` — default fields (no `model`),
   `models` = [A,B] → both "P/42" images returned, `total == 2`, hash equal on both items.
2. `test_compare_excludes_nonmatching_seed` — case 1's query does not include seed-43.
3. `test_compare_with_model_field` — add `fields=model` → only the reference's own model
   matches (`total == 1`).
4. `test_compare_model_filter_restricts` — `models=[A]` only → `total == 1`.
5. `test_compare_unknown_field_400`, `test_compare_missing_reference_404`.
6. `test_comparison_models_ordering_and_counts` — register a third model C via the API
   (never used) after A and B; response order is `[A, B, C, orphan]` (registered in added
   order, removed tail last); C has `image_count == 0`, `registered == true`; orphan entry
   has `registered == false`, count 1, snapshot name from its newest row; the empty-key row
   is excluded.
7. `test_compare_pagination` — `limit=1` returns `total==2` and one item; `offset=1`
   returns the other.
8. `test_compare_across_boards` — put A's image in board 1 and B's in board 2; result
   contains both with distinct `board_id` fields.

Run: `uv run pytest tests/unit/test_compare_hash.py tests/integration/test_api_compare.py`
then full `uv run pytest`. (Do not run ruff — repo policy.)

### 6.3 Manual verification

1. `uv sync && uv run uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload`
2. Install two models — call them A (first) and B (second) — and generate the same
   prompt/seed/cfg on each.
3. Select an image made by A → click ⇄ → selection = {A (anchor), B} (A is one of the last
   two added, so 2 checked); grid shows exactly the matched pair, badged with model names.
   Install model C → re-enter compare mode → C is auto-checked (selection {A, B, C}); C
   shows `(no images yet)` and adds no matches. Uncheck C → exit → re-enter → C stays
   unchecked (persisted; no longer "new"). Compare now on B's image (anchor B, unchanged
   set) → results identical except reference selection moves.
4. Toggle `seed` off in "Match by" → list grows (now matches every seed); toggle `model`
   on → list shrinks to one model's images.
5. Deselect all models → 0 images; Select all → all boards' matches.
6. Collapse/expand both groups; reload the page → field selections, model selections and
   group-collapse state survive; panel starts in browse mode; entering compare mode again
   shows the same checked set plus the new anchor (the currently selected image's model).
7. Enqueue a new matching generation while in compare mode → list refreshes within ~1 s.
8. Regression: browse mode behavior (board switch, search, jump nav, star/delete/move,
   context menu, arrow keys) unchanged.

## 7. Sequencing

1. `backend/gallery/compare.py` + unit tests (self-contained, defines the contract).
2. `GalleryService.model_image_stats` / `list_matching_images`.
3. Routes in `backend/api/images.py` + integration tests.
4. `README_API.md` entries.
5. `gallery.js` state/methods/branches; `index.html` markup; `style.css` classes.
6. Manual verification pass (§6.3).

Each step is independently testable; steps 1–3 land the backend before touching the UI.

## 8. Out of scope (explicit)

- No new DB columns/migrations; nothing written to the images table.
- No diff/side-by-side viewer of comparison results — clicking a result thumb uses the
  existing main-view display.
- `steps`, `sampler`, `prediction_type`, LoRAs, img2img parameters are not comparison
  fields (the field list is exactly the 7 specified).
- Comparison mode is not entered automatically by "Show in gallery" (`navigate-board`).
- No persisted per-board comparison history or hash column caching.
