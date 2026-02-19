# Test Suite Summary

## Attention & Mask Tests

| File | Tests |
|------|-------|
| `test_attention_mask_implementations.py` | Splash attention masks match model helpers; correct attention patterns for causal/teacher-forcing modes; multiplayer `block_size = spatial_size * num_players` |
| `test_kv_cache_analysis.py` | KV cache rolling buffer behavior; training/inference consistency; sliding window frames match between mask and cache |
| `test_splash_attention_outputs.py` | TPU splash attention kernels produce numerically correct outputs vs `jax.nn.dot_product_attention` |
| `test_mp_teacher_forcing_mask_bug.py` | Bug detection: confirms incorrect `block_size` in teacher-forcing produces wrong attention patterns |

## Data & Action Tests

| File | Tests |
|------|-------|
| `test_minecraft_mineflayer_actions.py` | Mineflayer action conversion: boolean keys, camera ranges |
| `test_minecraft_vpt_actions.py` | VPT action reading: binary keys, camera ranges, attack stuck handling, hotbar inheritance |
| `test_dataset_converters.py` | Dataset converters: `CameraLinearConverterMatrixGame2` output format |

## Configuration

| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures; `@requires_tpu` marker for TPU-only tests |
| `run_tests.sh` | Test runner script with `--no-tpu` and `--include-outdated` flags |

## Running Tests

```bash
./src/tests/run_tests.sh              # Run all tests
./src/tests/run_tests.sh --no-tpu     # Skip TPU tests
./src/tests/run_tests.sh -k "kv"      # Run tests matching "kv"
```
