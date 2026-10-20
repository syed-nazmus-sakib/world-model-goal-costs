# Visual evidence: when the goal image says more than the task

Created 2026-10-16 from the project's saved experiments. Open `index.html` for
the offline gallery. Every static figure is supplied as a PNG and an editable
SVG with embedded scene images. The Push-T replay is supplied as GIF and MP4.

## Suggested viewing order

1. `01_project_overview`: the existing planner, controlled interventions,
   measured goal-image failure, and unresolved research question.
2. `10_tested_goal_set_idea`: how the frozen-model goal-set experiment actually
   works, including its failed reversal gate and task-preservation controls.
3. `08_PushT_saved_plan_replay`: three saved action sequences replayed from the
   same initial state. Learned/latent and exact/latent fail; exact/true succeeds.
4. `03_*_equivalent_goals` and `09_Metaworld_*_goal_variants`: actual simulator
   images used in drawer-close and Metaworld-push goal-equivalence experiments.
5. `04_*_DINO_patch_cost`: measured DINOv2 feature discrepancies on those images.
6. `02_oracle_attribution_results`, `05_goal_set_paired_results`,
   `06_goal_set_failure_controls`, `07_same_candidates_different_rankings`:
   aggregate results, matched-episode outcomes, failure controls, and an
   illustrative candidate-ranking example.

## What is real, and what each visual means

- Environment images are actual saved simulator renders, not generated artwork.
- Push-T episode 002 was deliberately selected as an illustrative failure:
  the block is near its target but the pusher is misplaced. It is not an
  unbiased sample or a substitute for the aggregate 96-episode results.
- Replaying each saved plan reproduces the saved final state with maximum
  absolute difference **0.0**. See `source_data/replay_validation.json`.
- Push-T action denormalization uses the repository's precomputed dataset
  constants: mean `[-0.0087, 0.0068]`, std `[0.2019, 0.2002]`. Plans are reshaped
  from six grouped steps into 30 two-dimensional environment actions.
- For display, Push-T's decorative green target is aligned to the evaluation
  goal and a circle marks the goal pusher position. The physical replay is
  unchanged. The goal render can include the environment's normal settling step;
  success labels and displayed target coordinates use the saved evaluation state.
- Metaworld GIFs cycle counterfactual goals. They are **not temporal rollouts**,
  generated motion, or recordings of a successful controller. Variants 1–4
  share the checked object configuration; variant 0 is the expert goal, with
  small residual differences allowed by the original equivalence protocol.
- DINO maps use the same frozen DINOv2 ViT-S/14 backbone family as the evaluated
  JEPA-WM configuration, ImageNet input normalization, 224×224 RGB images,
  16×16 patches, and mean squared differences over 384 feature dimensions.
  They compare one fixed real scene (variant 2) to goals 0 and 3. Each pair shares
  a color scale. They exclude proprioception and are **cost-contribution maps,
  not attention maps, gradients, or causal saliency explanations**.
- Scatter plots and the pairwise flip matrix show one saved episode. Aggregate
  ranking/success claims come from the full evaluation, not that single example.
- The goal-set outcome figure uses the first 24 matched baseline episodes.
  The five-goal oracle contains the original goal. The mask is privileged
  robot-free simulator rendering; it is not a demonstrated deployable masking
  system for the learned planner.
- All goal-set closed-loop results displayed use exact simulator planning
  dynamics. The learned world model constructs the set; no claim is made that
  learned-dynamics planning has been validated by these results.
- There is **no validated general new method yet**. The current ungated set
  improves drawer variants but hurts the original goal, fails reach-wall, and
  loses to the masking baseline on drawer variant 3.

## Provenance and reproduction

`manifest.json` records captions and source locations. `source_data/` contains
the selected original episode files, measured DINO maps/features, and replay
frames/validation. Aggregate episode data remain in the project backup folders.
The supplied scripts do not alter existing experiments or launch new planning.

From the project root, with NumPy, Matplotlib, SciPy, and Pillow installed:

```bash
python scripts/visuals/make_visual_story.py
```

The MP4 conversion is optional:

```bash
ffmpeg -i results/visual_story_2026-10-16/08_PushT_saved_plan_replay.gif \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p \
  -movflags +faststart results/visual_story_2026-10-16/08_PushT_saved_plan_replay.mp4
python scripts/visuals/make_visual_story.py --gallery-only
```

`scripts/visuals/render_remote_assets.py` produces the raw replay and features
in the existing RunPod environment. It uses CPU DINO inference with cached
weights and requires the selected episode files. It neither trains a model nor
creates a pod. It exits if final-state replay differs by more than 0.01 in any
saved state coordinate; the delivered example matched exactly.

Figures summarize the scientific snapshot in `docs/07_RESTART_LOG.md` through
the 2026-10-16 go/no-go checkpoint. Later experiments are not automatically
incorporated. The gallery runs offline and makes no network requests.
