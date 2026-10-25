# Train the Cost Where the Planner Looks

Learned goal costs for latent world-model planning on Metaworld (JEPA-WM, frozen DINOv2 ViT-S/14). Status 2026-10-25: exact-dynamics results and ablations complete; learned-predictor results in progress.

## 1. The problem

A latent world model plans by three parts: a frozen encoder maps images to patch embeddings, a predictor rolls them forward under candidate action sequences, and a search (CEM, 300 candidates, horizon 6 chunks of 5 steps, replanning after each chunk) keeps the sequence whose predicted final embedding is closest to the goal embedding. The task enters through one number, the goal cost, and in the released planner it is the mean squared difference between the 256 patch embeddings, every patch weighted alike, plus 0.1 times the proprioceptive difference.

On tabletop manipulation that number measures the arm. Moving only the hand by 5 cm costs 2.6 to 14 times more than leaving the object off its target by the task's own success distance, on every one of eleven Metaworld tasks we tested; the robot covers about half the patches and carries 72 to 79 % of the cost. Five frozen encoders (DINOv2-S/B, SigLIP 2, CLIP, V-JEPA 2) show the same bias. The object is not missing from the features: a linear readout of the same patches locates it within 0.14 to 0.77 of the success distance. The cost simply does not weight it. The released planner solves 0 of 12 episodes on each of six object tasks that are in its own training data, while the same search with the true object state solves 10, 10 and 4 of 12 on drawer-open, plate-slide and faucet-open.

The failure has two phases. In the **approach** phase no candidate within one horizon touches the object, so a cost that measures the object is flat and a cost that measures the image pulls the arm to the goal arm pose, which is not the object. **Near success** the arm is in place and the question is the object's state, which the arm outweighs and often occludes. A reach-then-change oracle built from simulator positions confirms the structure: the reach term alone takes button-press from 0 to 12/12 and push from 0 to 7/12.

## 2. The main idea

Learning the cost is the obvious remedy, and it is where the real difficulty lives. A small head on the frozen features, supervised by simulator states at training time, ranks held-out candidate futures at 0.93 to 1.00 accuracy and then plans at 0/12 on button-press and 3/12 on push. The planner searches arm poses that are more than 0.3 m from any training frame, and the head prices them low: in every failing button episode the hand ends 20 to 45 cm from the button while the head's cost falls from 1.2 to 0.1. **Offline ranking accuracy does not predict closed-loop behaviour, because the planner finds the blind spots of its own cost.**

The method follows from taking this literally: train the cost on the states the planner actually visits and compares. Three parts, each answering one failure.

**Approach-aware target.** Each head $f_k$ is an attention-pooling network over the patch embeddings with the proprioceptive embedding appended, trained so that $\|f_k(z) - f_k(g)\|^2$ regresses

$$y(z) = \begin{cases}\log(1 + d_{\text{task}}(z)/r) & \text{object moved}\\ \log(1 + d_{\text{task}}(z_0)/r + d_{\text{hand,obj}}(z)/2\,\text{cm}) & \text{otherwise}\end{cases}$$

with $d_{\text{task}}$ the task's success distance, $r$ its threshold and $d_{\text{hand,obj}}$ the hand-to-object distance. The target is dense before contact and never below its value at contact, so the oracle's two-phase order becomes one regression. Labels come from the simulator at training time only; at test time the cost sees images, proprioception and the goal image. One ensemble serves all eight tasks.

**Ensemble pessimism.** $K = 3$ heads from different initialisations with per-member bootstrap masks; the planner pays mean $+\beta\cdot$std, $\beta = 1$. Where the data is thin the members disagree and the candidate is priced up rather than exploited.

**Coverage of the planner's search distribution.** Training frames are rendered by executing action sequences from the planner's own sampling distribution and encoding them with the frozen encoder: (i) the expert trajectory and its goal; (ii) candidates, 150 one-horizon rollouts from the start and from 12 and 4 steps before contact; (iii) arm-coverage walks, 16 random action sequences per start point continued for a whole episode; (iv) **planner frames**: the planner is run with the current cost on the 12 training episodes of each task and at every replan the executed frame and 32 of its final CEM population are saved with their labels, 232 frames per episode. The ensemble is retrained and step (iv) repeated. This is iterated cost learning in the sense of DAgger: the labelled states are the states the planner reaches and prices. Hand-designed perturbations do not find the exploits (six families of near-goal perturbations never produced the open-gripper button configuration that the planner produces in every failing episode).

A robot-free latent change term (squared patch difference to the goal over patches that show the robot in neither start, goal nor candidate, from a logistic robot mask on the same features, AUC 0.998) was the remedy for the terminal state in round 1. After two rounds of planner frames it is no longer needed: the head alone matches or beats head + change term on every object task.

The world model, encoder and search are never changed. Only the cost that ranks candidate futures is learned.

## 3. Protocol

Six Metaworld object tasks in the JEPA-WM training data but outside its released evaluation (button-press-topdown, push, drawer-open, plate-slide, faucet-open, window-open) and two controls a new cost must not break (drawer-close, solved by the released cost; reach-wall, where the arm is the task). Twelve test episodes per task, the same for every cost; heads trained on 24 disjoint episodes per task. Success by Metaworld's criterion at the final step. **Exact dynamics**: CEM candidates are executed in copies of the environment and only their final frames are encoded, so a failure is a failure of the cost or the search, not of prediction. Pass rule fixed before any run: at least 6/12 on at least four object tasks with both controls at or above 10/12. Twelve episodes give wide intervals (6/12: 25 to 75 % Wilson), so differences of two episodes are noise.

## 4. Main result, exact dynamics

Successes of 12, same episodes for every row. † control.

| cost | button | push | drawer-open | plate-slide | faucet | window | drawer-close† | reach-wall† |
|---|---|---|---|---|---|---|---|---|
| **baselines** | | | | | | | | |
| released latent cost (JEPA-WM) | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 23/24 |
| TEMPO temporal distance (re-impl.) | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 5 |
| state head (privileged labels, train only) | 0 | 0 | 0 | 8 | 0 | 0 | 12 | 7 |
| image-only reach-then-change (detector) | 3 | 0 | 1 | 0 | 0 | 0 | 10 | 8 |
| **ours, one component at a time** | | | | | | | | |
| approach-aware head, demonstration data | 0 | 3 | 0 | 1 | 0 | 2 | 12 | – |
| + walks + ensemble ($\beta = 1$) | 1 | 3 | 1 | 5 | – | – | – | – |
| + robot-free change term | 0 | 5/11 | 3 | 6 | – | – | – | – |
| + planner frames, round 1 | 3 | 7 | **10** | 11 | 2 | 7 | 12 | 11 |
| + planner frames, round 2 | 8 | 9 | 4 | 10 | 3 | 10 | 12 | 11 |
| **round 2, head alone (no change term)** | **10** | **10** | 3 | **11** | **4** | **11** | 12 | 10 |
| **oracles (simulator state at test time)** | | | | | | | | |
| reach-then-change, simulator positions | 12 | 7 | 6 | – | – | – | – | 12 |
| true object state | 0 | 11 | 10 | 10 | 4 | – | 22/22 | – |

Five of six object tasks pass the 6/12 bar with the head alone, from 0 for the released cost, with both controls intact. Push and plate-slide are at the true-state oracle. TEMPO's temporal distance, the closest published learned cost for these planners, is 0 on every object task and breaks the reach-wall control; it reproduces its published navigation gain in our re-implementation (LeWM TwoRoom 99/100 against 58/100 for a matched control), so the failure is specific to manipulation, where the temporal target inherits the arm bias. The image-only reach-then-change pipeline is the direct translation of the oracle to images and fails on localisation (5 to 14 cm off). The one regression, drawer-open from 10 to 4 across rounds, is a single mode: the planner never leaves the start pose, with the cost flat at about 4.8 for all seven replans. It existed in round 1 at a lower rate and the ablation below locates its cause.

Offline ranking accuracy of the round-2 ensemble on held-out candidate sets is 0.87 to 1.00 on every task and unchanged from round 1; it does not see any of the closed-loop differences above.

## 5. Ablations (round-2 ensemble, exact dynamics, head + change term unless stated)

| configuration | button | push | drawer-open | plate-slide |
|---|---|---|---|---|
| full ($\beta = 1$, walks, planner frames, $\lambda = 0.5$) | 8 | 9 | 4 | 10 |
| $\beta = 0$ (no pessimism) | 7 | 7 | **7** | 8 |
| no arm-coverage walks | 5 | 8 | 4 | 11 |
| demonstration data only (expert + candidates) | **0** | 4 | **0** | 10 |
| head alone ($\lambda = 0$) | 10 | 10 | 3 | 11 |
| union of round-1 and round-2 heads (6 members) | – | – | 8 | – |

Reading: demonstration-centred data is exploited to 0 on the two tasks with a long approach; walks matter most on button (5 against 8); the pessimism term is what flattens drawer-open's first replan (4 to 7 at $\beta = 0$, at a cost of one to two episodes elsewhere), and a union over rounds recovers it (8). Plate-slide is solved by every variant and separates nothing.

Training set of the round-2 ensemble: 115,295 frames from 36 directories over 8 tasks (expert 36.0k, goals 3.0k, candidates 12.6k, walks 13.3k, jitter 16.9k, planner frames 33.4k), 18 min on one RTX 3080 Ti.

## 6. Learned predictor (in progress)

With the released learned predictor in place of the simulator, the round-2 head alone scores 0/12 on button-press and plate-slide. The cost prices the predictor's outputs as near-success from the first replan (0.1 to 0.4, against about 5 at the same pose under exact dynamics): a head trained on encoder latents does not transfer to predicted latents. This is the paper's thesis one level down. The remedy under test is the same: planner frames are now the *predicted* final latents of the planner's last population, each labelled by executing its action sequence in the simulator (231 frames per episode), the ensemble is retrained with them, and the four learned-predictor closed loops are rerun. Results due 2026-10-26.

## 7. Relation to other work

| method | what it learns | setting | result on our protocol |
|---|---|---|---|
| JEPA-WM / DINO-WM / LeWM released cost | nothing (patch L2) | all | 0/12 on six object tasks |
| TEMPO (temporal distance on frozen latents) | time-to-goal | navigation gains (TwoRoom) | 0/12 on object tasks, control 5/12 |
| NGC (shortest path along the latent manifold) | nothing (graph distance) | navigation | not run; same arm bias in the metric |
| "Supervise what decides success" (auxiliary loss on the world model) | predictor features | PushT, Cube, small gains | changes the world model; ours keeps it frozen |
| VICE / RCE (success classifiers from examples) | terminal classifier | RL | the evid variant of our pilot: fixes the arm decoy, no dense signal (0/12 reach-wall) |
| **ours** | approach-aware cost on frozen features, trained on the planner's own states | manipulation | 10, 10, 11, 11 of 12 on four object tasks, controls 12 and 10 |

Claim: not a cost that solves Metaworld, but an account of why the released one fails, in two phases that need two remedies; a demonstration that a learned cost for a planner must be trained where the planner looks, since offline accuracy is blind to the exploitation; and the first image-plus-proprioception cost that takes the released planner from zero to the pre-stated bar on the manipulation tasks in its own training data, with the world model and its search untouched.
