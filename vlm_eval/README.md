# Minecraft Multiplayer VLM Evaluation Framework

A framework for evaluating video generation models on Minecraft multiplayer scenarios using Vision Language Models (VLMs). The evaluation measures whether generated videos correctly preserve multiplayer interactions.

---

## Quick start

### Environment Setup

```bash
cd vlm_eval/
conda env create -f environment.yml
conda activate solaris-eval

# Or, using venv
# python -m venv .venv
# source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set your Gemini API key (required for VLM evaluation, not needed for --dry-run / --extract-frames)
export GEMINI_API_KEY="your-api-key"
```

### Run evals

```bash
python run_all_evals.py
```

This scans `./../output` for model output folders (the output folder of the [inference script](../README.md#quick-gpu-inference)) and dataset in `./../datasets/eval`, and calls the underlying [run_eval.py](#run_evalpy) for every model output-dataset pair. Refer to the [run_all_evals.py](#run_all_evalspy) CLI section below for the list of CLI args.

### Extract frames (for debugging)

```bash
python run_all_evals.py --extract-frames
```

This scans `./../output` and `./../datasets`, and extracts all generated and ground truth frames that would be sent to the VLM for evaluation. It saves them in `./frame_extraction/` and `./frame_extraction_side_by_side/`.

## CLI

### `run_all_evals.py`

```bash
python run_all_evals.py --generations-dir GENERATIONS_DIR --dataset-base DATASETS_DIR [OPTIONS]
```

| Argument                  | Type / Default                                                                                                                                                       | Description                                                                                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--help`                  | flag                                                                                                                                                                 | Show help and exit.                                                                                                                                                                   |
| `--dry-run`               | flag (default: `False`)                                                                                                                                              | Print which `run_eval.py` commands would be executed without actually running them.                                                                                                   |
| `--extract-frames`        | flag (default: `False`)                                                                                                                                              | Extract frames only (no VLM queries). If `--generations-dir` has model folders, also writes side‑by‑side GT+generated comparison frames; otherwise extracts ground-truth frames only. |
| `--generations-dir PATH`  | `Path`, default `./../output`                                                                                                                                        | Directory containing model subfolders with generated videos.                                                                                                                          |
| `--dataset-base PATH`     | `Path`, default `./../datasets/eval`                                                                                                                                 | Base directory containing eval datasets (e.g. `translationEval`, `rotationEval`, ...).                                                                                                |
| `--results-dir PATH`      | `Path`, default `None`                                                                                                                                               | Root directory for JSON outputs; forwarded to `run_eval.py --results-dir` (falls back to `./results_json` when omitted).                                                              |
| `--limit INT`             | `int`, default `None`                                                                                                                                                | Limit number of episodes/queries per (model, dataset); passed through to `run_eval.py --limit`.                                                                                       |
| `--num-trials INT`        | `int`, default `1`                                                                                                                                                   | Number of evaluation trials per (model, dataset); forwarded to `run_eval.py --num-trials`. Must be ≥ 1.                                                                               |
| `-j`, `--max-workers INT` | `int`, default `min(num_models, CPU)`                                                                                                                                | Maximum number of models to evaluate in parallel. Use `1` for strictly sequential execution.                                                                                          |
| `--eval-types ...`        | space/comma list, default `ENABLED_EVAL_TYPES` (`translation`, `rotation`, `structure`, `turn_to_look`, `turn_to_look_opposite`, `one_looks_away`, `both_look_away`) | Which eval types to run. Accepts space- or comma-separated values and special value `all`. Unknown keys error out.                                                                    |
| `--models ...`            | space/comma list, default `None` (all models)                                                                                                                        | Filter to specific model folder names under `--generations-dir`. Accepts space- or comma-separated names and special value `all` (no filtering). Unknown names error out.             |

### `run_eval.py`

```bash
python run_eval.py DATASET_DIR --generated MODEL_OUTPUT_DIR [OPTIONS]
```

| Argument           | Type / Default                                    | Description                                                                                                                                                                                                                         |
| ------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `folder`           | positional (required)                             | Path to dataset folder (e.g. `datasets/eval/turnToLookEval`).                                                                                                                                                                       |
| `--help`           | flag                                              | Show help and exit.                                                                                                                                                                                                                 |
| `--dry-run`        | flag (default: `False`)                           | Print keyframe detection info per episode without extracting frames or querying the VLM. Useful for debugging handlers and dataset alignment.                                                                                       |
| `--extract-frames` | flag (default: `False`)                           | Extract frames only (no VLM queries). Frames are saved under `frame_extraction/{dataset}/{_real MODEL}/{QUERY_TYPE}/`. If `--generated` is set, also writes side-by-side GT+generated comparison frames.                            |
| `--output`, `-o`   | `str`, default `"eval_results.json"`              | Output path hint. By default, results are auto-organized into `results_json/generated/{model}_{dataset}/` or `results_json/real/{dataset}/`. If a `.json` path is provided, its suffix is stripped and treated as a directory name. |
| `--results-dir`    | `Path`, default `results_json/`                   | Root directory for auto-organized outputs when `--output` is left as the default.                                                                                                                                                   |
| `--limit INT`      | `int`, default `None`                             | Limit number of episodes to process. If omitted, a default cap of 32 video pairs is applied when more are available.                                                                                                                |
| `--num-trials INT` | `int`, default `1`                                | Number of full evaluation trials to run. Produces `trial_*.json` files plus an aggregate `stats.json` in the chosen output directory.                                                                                               |
| `--api-key STR`    | `str`, default `None`                             | Optional Gemini API key. If provided, sets `GEMINI_API_KEY` for this process; otherwise the `GEMINI_API_KEY` environment variable must already be set for non-`--extract-frames` runs.                                              |
| `--summary-json`   | `Path`, default auto-detected for structure evals | Path to `structure_building_summary.json` (only used for `structureEval` / `structureNoPlaceEval`). If omitted, a default path under `assets/hard_coded_gt/` is used.                                                               |
| `--generated PATH` | `Path`, default `None` (ground-truth only)        | Path to a generated-videos directory (e.g. `./../output/solaris/`). Enables evaluation of generated videos; also controls the model name used in output paths and frame extraction directories.                                     |

### Output Location

By default `run_eval.py` auto-organizes outputs under `results_json/`:

- **Ground-truth evaluations**: `results_json/real/{dataset_name}/trial_1.json` (plus `stats.json` summarizing `episode_level_accuracy.episode_accuracy` across trials).
- **Generated video evaluations**: `results_json/generated/{model_name}_{dataset_name}/trial_1.json` (plus `stats.json`).

You can override the root with `--results-dir` and the leaf directory name with `-o/--output` (see `run_eval.py --help`).

### Result Format

```json
{
  "vlm_model_name": "gemini-3-flash-preview",
  "our_model_name": "flagship_final",
  "thinking_enabled": false,
  "total_queries": 64,
  "correct": 58,
  "accuracy": 90.62,
  "vlm_errors_count": 0,
  "breakdown_by_query_type": {...},
  "episode_level_accuracy": {...},
  "results": [...],
  "vlm_errors": [...]
}
```

Each `trial_*.json` file has this shape; `stats.json` aggregates `episode_level_accuracy.episode_accuracy` across trials (mean, median, std, and per-trial values).

#### Episode-Level Accuracy

The `episode_level_accuracy` field measures whether ALL queries for each episode are correct (rather than individual query accuracy). This is particularly useful for multi-query datasets.

**For one player datasets** (e.g., `oneLooksAwayEval` with 2 query types per episode):

```json
"episode_level_accuracy": {
  "total_episodes": 32,
  "fully_correct_episodes": 28,
  "episode_accuracy": 87.5,
  "is_both_players_dataset": false
}
```

**For both players datasets** (e.g., `bothLookAwayEval` with 2 query types × 2 players = 4 queries per episode):

```json
"episode_level_accuracy": {
  "total_episodes": 32,
  "fully_correct_episodes": 24,
  "episode_accuracy": 75.0,
  "is_both_players_dataset": true,
  "per_player_episode_accuracy": {
    "alpha": {
      "total_episodes": 32,
      "fully_correct_episodes": 28,
      "episode_accuracy": 87.5
    },
    "bravo": {
      "total_episodes": 32,
      "fully_correct_episodes": 26,
      "episode_accuracy": 81.25
    }
  }
}
```

| Field                         | Description                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `episode_accuracy`            | % of episodes where ALL queries are correct                                     |
| `per_player_episode_accuracy` | (Both-players only) % of episodes where all queries for that player are correct |

## Implementation

### Available Datasets

The codebase supports the following eval datasets with every datasets having a dedicated [handler](#handler-architecture).

| Dataset                  | Description                             | Queries/Episode               | Expected Answer                     |
| ------------------------ | --------------------------------------- | ----------------------------- | ----------------------------------- |
| `turnToLookEval`         | Both players turn to look at each other | 1                             | yes                                 |
| `turnToLookOppositeEval` | Players do NOT look at each other       | 1                             | no                                  |
| `translationEval`        | Detects player movement direction       | 2 (both perspectives)         | closer/farther/left/right/no motion |
| `rotationEval`           | Detects camera rotation direction       | 1                             | left/right/ no player               |
| `oneLooksAwayEval`       | One player looks away and back          | 2 (presence queries)          | yes/no                              |
| `bothLookAwayEval`       | Both players look away and back         | 4 (2 bots × presence queries) | yes/no                              |
| `structureEval`          | Structure is built and visible          | 1                             | yes                                 |

### Code Flow Overview

When you run:

```bash
python run_eval.py DATASET_DIR --generated MODEL_OUTPUT_DIR
```

The execution flows through these steps:

#### 1. Argument Parsing (`main()`)

- Parses command-line arguments
- Extracts the dataset name from the path for handler identification

#### 2. Handler Identification (`identify_handler()`)

- Matches the dataset name against each handler's `DATASET_NAMES` attribute
- Returns an instance of the appropriate handler class
- For structure handlers (`structureEval`, `structureNoPlaceEval`), automatically loads the required ground-truth summary JSON from `assets/hard_coded_gt/`

#### 3. Video Pair Discovery (`find_mc_video_pairs()`)

- Scans the dataset folder for video files matching the pattern:

  ```
  {episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
  ```

- Groups matching files into `VideoPair` objects containing:
  - `alpha_video`, `bravo_video`: Paths to the video files
  - `alpha_json`, `bravo_json`: Paths to corresponding metadata JSON files
  - `episode_num`, `instance_num`: Identifiers for the episode
- Returns pairs sorted by episode and instance number
- **Default limit**: 32 video pairs (override with `--limit`)

#### 4. Generated Video Matching (if `--generated` is provided)

- Calls `find_generated_video_subdir()` to locate the correct subdirectory within the generations folder
- Generated videos are named `video_{N}_side_by_side.mp4` where N corresponds to the video pair index
- The generated video is a 1280x720 composite with 4 quadrants:
  - **Top-left**: Alpha ground-truth
  - **Top-right**: Alpha generated
  - **Bottom-left**: Bravo ground-truth
  - **Bottom-right**: Bravo generated

#### 5. Keyframe Extraction (`handler.extract_keyframes()`)

- Each handler implements its own keyframe extraction logic
- Reads the JSON metadata files to determine:
  - Which frames contain the relevant action/event
  - Which bot's perspective to evaluate from
  - What the expected answer should be
- Returns a list of `KeyframeQuery` objects with:
  - `video_path`: Which video to extract from
  - `frame_index`: Which frame to evaluate
  - `expected_answer`: The correct answer for validation
  - `metadata`: Additional context (variant, frame indices, etc.)

#### 6. VLM Evaluation Loop (`run_evaluation()`)

For each keyframe query:

1. **Extract frame(s)** using `extract_frame()` or `extract_frame_from_generated()`
   - Ground-truth: Extracts directly from the video file
   - Generated: Extracts from the appropriate quadrant (top-right for alpha, bottom-right for bravo)
2. **Query VLM** using `query_vlm()` with the handler's prompt and extracted frame(s)
3. **Validate response** using `handler.validate_response()`
4. **Record result** with correctness and metadata

#### 7. Result Saving (`save_results()`)

- Saves comprehensive JSON with:
  - Overall statistics (total queries, correct count, accuracy)
  - Per-query breakdown (query type if applicable)
  - Episode-level accuracy (% of episodes where ALL queries are correct)
  - Per-player episode accuracy for both-players datasets
  - Individual result details
- Auto-organized output paths:
  - Ground-truth: `results_json/real/{dataset_name}/`
  - Generated: `results_json/generated/{model_name}_{dataset_name}/`

### Handler Architecture

Each handler extends `EpisodeTypeHandler` and implements:

| Method/Attribute                        | Description                                                             |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `DATASET_NAMES`                         | List of exact dataset folder names this handler supports                |
| `get_prompt()`                          | Returns the VLM prompt for evaluation                                   |
| `extract_keyframes(video_pair)`         | Determines which frames to evaluate and returns `KeyframeQuery` objects |
| `validate_response(response, expected)` | Compares VLM response to expected answer                                |
| `enable_vlm_thinking`                   | Whether to enable VLM thinking mode (default: False)                    |

Available handlers:

- `MinecraftTurnToLookHandler`
- `MinecraftTurnToLookOppositeHandler`
- `MinecraftTranslationHandler`
- `MinecraftRotationHandler`
- `MinecraftLooksAwayHandler`
- `MinecraftBothLookAwayHandler`
- `MinecraftStructureBuildingHandler`

### Structure Evaluation: Special Requirements

The `structureEval` and `structureNoPlaceEval` datasets require hard-coded ground-truth files because the structure type (wall, tower, square) and which bot builds are **randomly selected during data generation**.

### How Structure GT Works

1. **The JSON format** maps instance → episode → builder/structure info:

   ```json
   {
     "instance_0": {
       "episode_0": {
         "builder": "alpha",
         "structure": "wall_4x1",
         "alpha_structure": "wall_4x1",
         "alpha_builds": true,
         "bravo_structure": "wall_4x1",
         "bravo_builds": false
       }
     }
   }
   ```

2. **During evaluation**, the structure handler:
   - Looks up which bot is the observer (non-builder)
   - Extracts frames from the observer's perspective
   - Asks the VLM if a structure is visible

#### Episode Order Consistency

The ground-truth JSON files assume episodes are processed in **sorted order by episode number and instance**. If the episodes in your dataset are shuffled or reordered, the structure type lookups will be incorrect. The `find_mc_video_pairs()` function ensures this by returning pairs sorted by `(episode_num, instance_num)`.

The script automatically detects the eval type from log contents and outputs to `assets/hard_coded_gt/`.

### Video Format

#### Ground-Truth Videos

```
{episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4  # Video file
{episode}_{Alpha|Bravo}_instance_{instance}.json        # Metadata (frame-by-frame state)
```

#### Generated Videos (Side-by-Side)

```
video_{N}_side_by_side.mp4   # 1280x720, 4 quadrants
```

Layout:

```
┌─────────────────┬──────────────────┐
│  Alpha GT       │  Alpha Generated │
│  (top-left)     │  (top-right)     │
├─────────────────┼──────────────────┤
│  Bravo GT       │  Bravo Generated │
│  (bottom-left)  │  (bottom-right)  │
└─────────────────┴──────────────────┘
```
