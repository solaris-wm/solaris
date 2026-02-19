# Minecraft Multiplayer VLM Evaluation Framework

A framework for evaluating video generation models on Minecraft multiplayer scenarios using Vision Language Models (VLMs). The evaluation measures whether generated videos correctly preserve multiplayer interactions that were present in the ground-truth training data.

---

# Part 1: Usage

## Quick Start

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key"



```

## Environment Setup

```bash
# (Recommended) Create and activate a dedicated environment
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

## Command Reference

```text
python run_eval.py <folder> [options]

Arguments:
  folder                    Path to dataset folder (e.g., datasets/eval/turnToLookEval)
                            The /test subdirectory is automatically appended.

Modes (mutually exclusive):
  --dry-run                 Print keyframe metadata only (no frames, no VLM calls, no API key needed)
  --extract-frames          Extract frames only (no VLM calls). Frames are saved under frame_extraction/... and,
                            if --generated is set, frame_extraction_side_by_side/...

Options:
  --generated PATH          Path to generated videos directory
  --limit N                 Limit number of episodes to process (default: 32 video pairs)
  --num-trials N            Number of evaluation trials to run (default: 1). Writes trial_*.json plus stats.json.
  --results-dir PATH        Root directory for auto-organized JSON outputs (default: results_json)
  --api-key KEY             Gemini API key (or use GEMINI_API_KEY env var)
  --summary-json PATH       Custom path to structure summary JSON (structure datasets only)
  -o, --output PATH         Output directory or legacy .json path (see Output Location section)
```

## Available Datasets

| Dataset                  | Description                             | Queries/Episode               | Expected Answer                     |
| ------------------------ | --------------------------------------- | ----------------------------- | ----------------------------------- |
| `turnToLookEval`         | Both players turn to look at each other | 1                             | yes                                 |
| `turnToLookOppositeEval` | Players do NOT look at each other       | 1                             | no                                  |
| `translationEval`        | Detects player movement direction       | 2 (both perspectives)         | closer/farther/left/right/no motion |
| `rotationEval`           | Detects camera rotation direction       | 1                             | left/right/ no player               |
| `oneLooksAwayEval`       | One player looks away and back          | 2 (presence queries)          | yes/no                              |
| `bothLookAwayEval`       | Both players look away and back         | 4 (2 bots × presence queries) | yes/no                              |
| `structureEval`          | Structure is built and visible          | 1                             | yes                                 |

**Note**: For datasets with multiple queries per episode, the `episode_level_accuracy` metric in results shows what percentage of episodes have ALL queries correct.

## Common Usage Examples

### Dry Run (Inspect Without VLM Queries)

```bash
# View keyframe detection info for first 10 episodes (no API cost)
python run_eval.py ./datasets/eval/turnToLookEval --dry-run --limit 10
```

### Extract Frames for Visual Inspection

```bash
# Extract frames to frame_extraction/ folder
python run_eval.py ./datasets/eval/turnToLookEval --extract-frames --limit 5
```

### Run Full Evaluation

```bash
# Evaluate ground-truth videos (sanity check)
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval

# Evaluate generated videos
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final
```

### Batch Evaluation (All Models)

```bash
python run_all_evals.py
```

This scans `generations/` for model folders and runs evaluations for each enabled dataset.

- **Configurable eval types**: edit `ENABLED_EVAL_TYPES` in `run_all_evals.py` or pass `--eval-types ...` / `--eval-types all`.
- **Model selection**: use `--models MODEL_A MODEL_B` to restrict which subdirectories under `generations/` are evaluated.
- **Trials and outputs**: forward `--num-trials`, `--limit`, and `--results-dir` to control the number of trials and output locations.
- **Frame extraction only**: pass `--extract-frames` (optionally with `--dry-run`) to just dump frames (and GT vs generated side-by-side images) without querying the VLM.

## Output Location

By default `run_eval.py` auto-organizes outputs under `results_json/`:

- **Ground-truth evaluations**: `results_json/real/{dataset_name}/trial_1.json` (plus `stats.json` summarizing `episode_level_accuracy.episode_accuracy` across trials).
- **Generated video evaluations**: `results_json/generated/{model_name}_{dataset_name}/trial_1.json` (plus `stats.json`).

You can override the root with `--results-dir` and the leaf directory name with `-o/--output` (see `run_eval.py --help`).

## Result Format

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
  "vlm_errors": [...]  // only present if some queries failed due to API errors
}
```

Each `trial_*.json` file has this shape; `stats.json` aggregates `episode_level_accuracy.episode_accuracy` across trials (mean, median, std, and per-trial values).

### Episode-Level Accuracy

The `episode_level_accuracy` field measures whether ALL queries for each episode are correct (rather than individual query accuracy). This is particularly useful for multi-query datasets.

**For single-player datasets** (e.g., `oneLooksAwayEval` with 3 query types per episode):

```json
"episode_level_accuracy": {
  "total_episodes": 32,
  "fully_correct_episodes": 28,
  "episode_accuracy": 87.5,
  "is_both_players_dataset": false
}
```

**For both-players datasets** (e.g., `bothLookAwayEval` with 3 query types × 2 players = 6 queries per episode):

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

---

# Part 2: Implementation

## Code Flow Overview

When you run:

```bash
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final --limit 10
```

The execution flows through these steps:

### 1. Argument Parsing (`main()`)

- Parses command-line arguments
- Automatically appends `/test` to the dataset path (e.g., `turnToLookEval` → `turnToLookEval/test`)
- Extracts the dataset name from the path for handler identification

### 2. Handler Identification (`identify_handler()`)

- Matches the dataset name against each handler's `DATASET_NAMES` attribute
- Returns an instance of the appropriate handler class
- For structure handlers (`structureEval`, `structureNoPlaceEval`), automatically loads the required ground-truth summary JSON from `assets/hard_coded_gt/`

### 3. Video Pair Discovery (`find_mc_video_pairs()`)

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

### 4. Generated Video Matching (if `--generated` is provided)

- Calls `find_generated_video_subdir()` to locate the correct subdirectory within the generations folder
- Generated videos are named `video_{N}_side_by_side.mp4` where N corresponds to the video pair index
- The generated video is a 1280x720 composite with 4 quadrants:
  - **Top-left**: Alpha ground-truth
  - **Top-right**: Alpha generated
  - **Bottom-left**: Bravo ground-truth
  - **Bottom-right**: Bravo generated

### 5. Keyframe Extraction (`handler.extract_keyframes()`)

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

### 6. VLM Evaluation Loop (`run_evaluation()`)

For each keyframe query:

1. **Extract frame(s)** using `extract_frame()` or `extract_frame_from_generated()`
   - Ground-truth: Extracts directly from the video file
   - Generated: Extracts from the appropriate quadrant (top-right for alpha, bottom-right for bravo)
2. **Query VLM** using `query_vlm()` with the handler's prompt and extracted frame(s)
3. **Validate response** using `handler.validate_response()`
4. **Record result** with correctness and metadata

### 7. Result Saving (`save_results()`)

- Saves comprehensive JSON with:
  - Overall statistics (total queries, correct count, accuracy)
  - Per-query breakdown (query type if applicable)
  - Episode-level accuracy (% of episodes where ALL queries are correct)
  - Per-player episode accuracy for both-players datasets
  - Individual result details
- Auto-organized output paths:
  - Ground-truth: `results_json/real/{dataset_name}/`
  - Generated: `results_json/generated/{model_name}_{dataset_name}/`

## Handler Architecture

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
- `MinecraftStructureNoPlaceHandler`

## Structure Evaluation: Special Requirements

The `structureEval` and `structureNoPlaceEval` datasets require hard-coded ground-truth files because the structure type (wall, tower, square) and which bot builds are **randomly selected during data generation**.

### How Structure GT Works

1. **During data collection**, logs are generated that record:
   - Which bot (Alpha or Bravo) performed the building action
   - What structure type was randomly selected (wall_4x1, tower_2x1, wall_2x2)

2. **`parse_structure_logs.py`** parses these logs to create:
   - `assets/hard_coded_gt/structure_building_summary.json` (for structureEval)

3. **The JSON format** maps instance → episode → builder/structure info:

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

4. **During evaluation**, the structure handler:
   - Looks up which bot is the observer (non-builder)
   - Extracts frames from the observer's perspective
   - Asks the VLM if a structure is visible

### Important: Episode Order Consistency

The ground-truth JSON files assume episodes are processed in **sorted order by episode number and instance**. If the episodes in your dataset are shuffled or reordered, the structure type lookups will be incorrect. The `find_mc_video_pairs()` function ensures this by returning pairs sorted by `(episode_num, instance_num)`.

### Regenerating Structure GT Files

If you need to regenerate the structure ground-truth files from new logs:

```bash
# For structureEval
python parse_structure_logs.py /path/to/structureEval/logs_directory

# For structureNoPlaceEval
python parse_structure_logs.py /path/to/structureNoPlaceEval/logs_directory
```

The script automatically detects the eval type from log contents and outputs to `assets/hard_coded_gt/`.

## Video Format

### Ground-Truth Videos

```
{episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4  # Video file
{episode}_{Alpha|Bravo}_instance_{instance}.json        # Metadata (frame-by-frame state)
```

### Generated Videos (Side-by-Side)

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

## Project Structure

```
├── environment.yml              # Conda environment (Python 3.10 base)
├── requirements.txt             # Python dependencies (google-genai, opencv-python-headless, ...)
├── run_eval.py                  # Main entry point for all evaluations
├── vlm_utils.py                 # Core utilities (VLM queries, frame extraction, data classes)
├── run_all_evals.py             # Batch evaluation across all models
├── parse_structure_logs.py      # Parses structure building logs to create GT files
├── visualization_helper.py      # Helpers for GT vs generated side-by-side visualizations
├── extract_all_frames.py        # Utility to dump frames across all tasks (debugging)
│
├── handlers/                    # Episode type handlers
│   ├── __init__.py
│   ├── mc_multiplayer_handler_translation.py
│   ├── mc_multiplayer_handler_rotation.py
│   ├── mc_multiplayer_handler_looks_away.py
│   ├── mc_multiplayer_handler_both_look_away.py
│   ├── mc_multiplayer_handler_structure.py
│   ├── mc_multiplayer_handler_turn_to_look.py
│   ├── mc_multiplayer_handler_turn_to_look_opposite.py
│   └── camera_utils.py
│
├── assets/
│   └── hard_coded_gt/           # Hard-coded ground truth for structure evaluations
│       ├── structure_building_summary.json

```
