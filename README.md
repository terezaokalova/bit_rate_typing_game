# Bit Rate Game

## Start

```
./run.sh
# Opens at http://localhost:8080
# Calibration: ~25 seconds (three 7-second rounds plus transitions)
# Scored run: 60 seconds at your personalized N and alphabet
```

Requirements are Python 3 and a web browser. Modes available are default (calibration plus scored run), sandbox (`?n=18&duration=30`), data collection (`?mode=collect`), and a chord experiment (`?mode=chord`).

## The problem

The achieved bit rate formula from Shenoy et al. (2021):

```
B = log₂(N - 1) × max(Sc - Si, 0) / t
```

The max(Sc - Si, 0) structure is the constraint that shapes everything downstream. Each error cancels a correct selection, making errors doubly costly. The effective accuracy factor (2p - 1) hits zero at 50% accuracy. At 95% it is 0.90, and at 75% it drops to 0.50, halving throughput. The penalty structure shifts the optimal operating point hard toward accuracy.

## Why keyboard input

Keyboard is among the highest-bandwidth trained motor skills humans have for this kind of task. Dhakal et al. (2018) report a mean of 51.56 WPM (SD 20.20) across 168,000 volunteers and 136 million keystrokes, the largest typing study to date. Hick's law and Hyman's law (Hick, 1952; Hyman, 1953) describe how reaction time grows with the log of the alphabet size in choice tasks, but the slope depends on stimulus-response compatibility. Leonard (1959) showed that when vibrotactile stimulation was delivered to a fingertip and the response was to depress the vibrating armature with the stimulated finger, RT was statistically flat across 2, 4, and 8 alternatives, collapsing the Hick-Hyman slope to zero. I considered spatial click targets, but Fitts' law (Fitts, 1954) governs aimed-movement throughput, and the survey by Soukoreff and MacKenzie (2004) of ISO 9241-9 evaluations places mouse throughput in the 3.7 to 4.9 bps range, with direct finger touch on a smartphone reaching 6.95 bps (MacKenzie, 2015). Both sit below the per-keystroke information rate achievable with trained QWERTY typing, where the motor mapping is overlearned and stimulus-response compatibility is high.

## Why adaptive N

The spec says "alphabet of size N >= 3" with no constraint that N be the same for all players. The rest of the calibration, pruning, and scoring logic flows from this design choice.

Borrowing from the adaptive modulation problem in digital communications, the transmitter measures channel quality (calibration) and selects the highest-order constellation (largest N) the channel will support without the error rate destroying throughput. QPSK, 16-QAM, and 64-QAM each pack more bits per symbol but require a cleaner channel to decode reliably. Here, calibration measures each player's channel quality and selects their optimal constellation.

Before building the game, I ran a Monte Carlo simulation (`analysis/simulation.py`) sweeping N from 3 to 40 across stochastic agent profiles under two accuracy models. The linear and logistic models disagreed on N* by roughly a factor of 8 for the same agent. Seeing that gap, I stopped trying to pick N based on theoretical modeling alone and built a calibration system so that I could measure each player's accuracy curve.

## Optimizing N: the derivation

Bit rate combines three factors:

```
B(N) = log₂(N - 1) × R(N) × (2p(N) - 1)
```

The Hick-Hyman law gives R(N) = 1 / (a + b × log₂(N)), where a is base sensorimotor latency (comparable to intrinsic BCI delay) and b is per-bit processing cost. Letting x = log₂(N) and using a linear accuracy model p(x) = p₀ - cx, with α = 2p₀ - 1 and β = 2c, the bit rate becomes:

```
B(x) = x × (α - βx) / (a + bx)
```

Setting dB/dx = 0 yields βb·x² + 2βa·x - αa = 0, with positive root x* = [-βa + √(βa(βa + bα))] / (βb), and N* = 2^(x*). The optimal N depends on each player's individual parameters (a, b, p₀, c), which is why calibration measures them empirically.

## Per-key optimization and greedy pruning

Not all keys are equal. Home row keys require zero hand movement, SPACE is thumb-only, and edge keys requiring pinky-off-home-row movement (Q, P, Z, ', /, ., ,) are slower and less accurate.

Each key gets a composite score, `score = mean_RT × (1 + 3 × error_rate)`. The factor of 3 sits above the Shenoy formula's implicit 2× penalty (each error cancels two correct selections), with the extra unit reflecting that error rates increase under scored-run pressure relative to calibration. A factor of 2 matched Shenoy exactly, and a factor of 4 led the pruner to exclude too many keys for fast users.

Starting from the model-recommended N*, the pruner runs a greedy loop. It identifies the worst key by composite score and checks whether removing it improves predicted bit rate. If `B(N-1, without worst) >= B(N, with worst)`, the key is dropped and the loop continues; otherwise it short-circuits. The 2³¹ possible alphabet subsets make exhaustive search intractable, so greedy with the per-key composite score is a tractable approximation, not provably optimal.

**Why the priority tiers split where they do:** the order is grouped by required finger movement, not alphabet position. Tier 1 (N=8) is the eight home-row resting positions A S D F J K L semicolon, where pressing them requires zero movement. Tier 2 adds SPACE (thumb, no displacement). Tier 3 adds G and H, which require lateral index-finger movement of one position inward. The final tier adds true edge keys requiring pinky-off-home-row movement. Calibration round 3 samples only this final tier, because these are where players differ most.

## Alternatives considered and rejected

**Chording:** I wanted to try this with the rationale being that if you press two (or multiple) keys simultaneously instead of one at a time, you considerably expand the symbol space without moving the fingers off the home row. In particular, eight home-row keys give 28 same-hand pairs, which is log₂(27) = 4.75 bits per selection, so I reasoned that packing more bits into each motor act might would single-key input outright. I implemented it as `?mode=chord`and tested it on myself. However, it turned out the 80ms chord detection window plus the ~200ms of extra coordination overhead led to ~2.69 bps versus 9+ bps in letter mode (at that point of my development pipeline). It turns out that the break-even condition requires chord RT below single-key RT × 0.91, which I never came even close to hitting. The coordination cost was way larger than the added value of the extra bits. I left it in as an experimental mode only with the break-even math shown.

**Digraph and tuple targets:** I considered a 26²-symbol digraph alphabet, which would carry 9.7 bits per selection. But players type the second character of a known digraph faster than the first, so within-symbol RT is non-uniform and the i.i.d. assumption would break, thereby violating the spec. 

**Alphanumerics (letters plus digits 0–9).** Tested empirically. Adding the digit row expands N to 36, raising per-selection information from log₂(25) = 4.64 bits to log₂(35) = 5.13 bits. But measured cost was disqualifying. Digit RTs ran 850–1200ms versus 250–350ms for letters, a 3× slowdown attributable to the digit row sitting one full row above home position and being a much less practiced motor target. My pilot runs gave 9.06 bps at N=36 versus 10.53 bps at N=26 on the same player. Consequently, the information gain did not compensate for the speed cost. One trial also showed an confusion between the letter O and digit 0, which becomes riskier with time pressure.

**Speech and vocalization input (phonemes or alphanumerics as symbols).** Considered as a non-keyboard modality, but decided against it. First, browser speech recognition (Web Speech API) carries a floor latency of 300–500ms per token, already above the per-selection budget that keyboard typing achieves. Second, isolated-phoneme accuracy in quiet conditions sits at roughly 70–80%, which, under the Shenoy penalty structure (2p − 1), kill half the throughput before any other cost is paid. Third, microphone permissions, ambient noise, and hardware variance across three graders' machines would make the channel non-stationary in a way the calibration protocol cannot correct for. 

**Spatial-pattern chord display (dot-grid stimulus, finger-mirror response).** Also considered variant of chord mode in which the stimulus is a 2×4 dot pattern and the response is to press the keys whose positions match the lit dots. This would push Leonard (1959) to its limit, where the stimulus is the response and stimulus-response compatibility is maximized. Decided against it as, even though the motor mapping is theoretically optimal, the symbol set is unfamiliar, and graders would need to learn 20 spatial patterns during a brief familiarization period. Contrastingly, letters are learned over years of typing practice. Moreover, the same coordination overhead that caused issues in the standard chord mode (~200ms per selection) would apply.

**Modifier and edge keys (Tab, Shift, Caps Lock, Cmd, Option).** Considered as a way to expand N beyond the 31 standard alphanumeric keys. Rejected because modifier keys carry browser-level side effects (Tab moves focus, Cmd triggers OS shortcuts), have irregular event handling across operating systems, and have non-uniform physical sizes that would break the visual symmetry of the keyboard diagram.

**Visible gamification layer (progress bar, neuron-firing animation, failure feedback).** Considered for engagement, including a cumulative-progress visualization or a tripping-dinosaur-style (think offline Google) visual cue whenever an error occurs. Decided against it: the lookahead queue already competes for visual attention with the current target, and earlier testing showed that any element drawing focus away from the queue costs throughput substantially (a roughly 4× drop in bit rate when queue prominence was reduced). Adding a second visual channel would either be ignored (no benefit) or attended to (direct throughput cost). The audio cue on each new target is the only feedback channel that runs in parallel with the visual stream without competing for it.

## Implementation

**Architecture:** single-page HTML, JavaScript, and CSS. No build step, no dependencies, no framework. Graders clone, run ./run.sh, and play within five seconds, with nothing to install and no version drift between runs. Internally the code is organized into isolated sections (calibration, scoring, rendering, information-theoretic analysis), each wrapped in its own IIFE scope to enforce encapsulation and prevent cross-section state leakage. I'm one person shipping a 60-second game, so I wanted the section boundaries enforced but the file still readable front-to-back. At this scope, I thought that a bundler or ES module graph would add setup without much benefit at this scope.

**Module boundaries:** `KeyNormalizer` translates browser key events into the canonical alphabet. `KeySubsetSelector` owns the priority tiers and selects the active alphabet for any N. `KeyboardDiagram` renders the visual keyboard. `CalibrationFitter` runs the OLS Hick-Hyman fit, the per-key composite scoring, and the greedy pruner. `BitRateCalculator` evaluates the Shenoy formula and the mutual information throughput. `DataExporter` serializes session JSON. The main game loop is a state machine with named states (welcome, calibrating, scored, results) and a freeze-state interrupt for phase-shift recovery.

**Verification:** three independent layers cross-check the implementation. The Monte Carlo simulation in analysis/simulation.py served two purposes: sweeping N across accuracy models showed that N* is hypersensitive to model choice (motivating the empirical calibration thereafter), and for any fixed accuracy model the simulated argmax B(N) agrees with the analytic optimum from the quadratic root (a numerical check on the closed-form derivation). The quantitative results reported throughout this README were independently recomputed in Python from raw session JSONs. First-session user data served as an integration test on the calibration such that when calibration recommends N* but the player's scored run produces a bit rate substantially lower than predicted, that is a regression signal. The slow-typist case documented in the known limitations section was surfaced this way, and remains unresolved because the short calibration rounds do not elicit the error rates that may appear under scored-run pressure.

**Lookahead queue:** four upcoming targets visible alongside the current target, with graded size and opacity. Motor preparation for trial N+1 overlaps with stimulus identification for trial N, converting serial latency into pipelined throughput. The measured pipelining gain was ~60% (69 to 114 selections per 60s). I tested depths of 3, 4, and 5; 4 felt right and was kept without a formal sweep. The bigger tradeoff was visual prominence. During my own runs, I noticed I was constantly looking ahead in the queue, which is the very behavior the queue exists to enable. But when I tried making the queue more subtle to push focus back onto the current target, my bit rate dropped roughly four-fold because I was no longer registering the upcoming key in time. The setting that worked was making the current letter visibly larger and more salient than the queued items while keeping the queue itself fully visible.

**Phase-shift recovery:**  the lookahead queue creates a specific failure mode. During one of my own runs I caught myself responding to the next target instead of the current one for over ten consecutive trials, because my gaze had locked onto the queue. To prevent this behavior, if it so happens that three consecutive errors each match the next target (the player is responding one-ahead), the system freezes input for 500ms and pulses the current target. This is inspired by frame synchronization in digital communications, where a receiver that has lost symbol alignment pauses and resynchronizes before resuming.

**Spatial audio cues:** a brief sine tone on each new target indicates the target's keyboard zone (left 260 Hz, center 440 Hz, right 740 Hz). The bands were chosen to be obviously discriminable to a first-time listener, not to extract maximum auditory information; a perceptually uniform spacing (e.g., equal steps on the mel scale) would be more principled if the audio were the primary channel. I also considered using two tones to signal correct versus incorrect on the previous selection, but I dropped that idea because backward-looking feedback does not help throughput. The forward-looking anticipatory cue is the one that actually has a chance to help. Some users reported that they did not consciously notice or use the audio, so the practical benefit is modest.

**Input handling:** keydown events with event.repeat filtering, performance.now() for all timing, and requestAnimationFrame for display updates. Keys outside the active alphabet flash invalid without changing Sc or Si.

## Calibration protocol

Three 7-second rounds. Rounds 1 and 2 use general keys at N=16 and N=26; round 3 uses a dedicated edge-key set at N=12. From rounds 1 and 2 only, the Hick-Hyman law and a linear accuracy model are fit using OLS in transformed log-space. With two data points the fit is exact. Round 3 would bias the estimates with systematically slower keys. N* is found by sweeping N from 3 to 31 and selecting argmax B(N).

**Fitting the accuracy model.** The accuracy model p(N) = p₀ - c·log₂(N) is fit two ways; the recommendation uses the OLS result, with a Bayesian fit logged alongside for cross-validation. When the two anchors produce a non-negative slope (accuracy flat or improving with N, which contradicts the decay assumption and usually reflects noise), c is set to 0 and p₀ to the mean observed accuracy. When the slope is negative (the expected case of accuracy decay with larger alphabets), c and p₀ are taken directly from the OLS solution. Both parameters update together so the fitted line stays consistent with the calibration anchors.

**Bayesian cross-check.** OLS on two anchor points with ~20 trials each is sensitive to single-trial noise. I noticed during pilot testing that a single error in round 2 could shift c by ~0.05 and swing N* by 10 or more positions between back-to-back calibrations on the same player. That was unacceptable to me for a system whose entire output is a single integer recommendation, so I added a Bayesian linear regression as a second estimator to flag cases where the OLS fit was being pulled around by noise. I chose a weakly informative prior (p₀ ~ N(1.0, 0.1²), c ~ N(0.02, 0.03²)) to encode my expectation that players are generally accurate and accuracy decays slowly with N, without overriding genuine signal when the data is clean. The per-anchor likelihood is binomial-Gaussian with σ_i² = p̂(1-p̂)/n + 1/(4n); I added the +1/(4n) continuity correction after hitting divide-by-zero errors when an anchor returned 100% accuracy. The Bayesian posterior is logged to the console and exported in the session JSON for each calibration. When OLS and Bayesian fits agree on N*, I treat confidence as high; when they disagree, I trust the Bayesian fit as the more conservative answer because it shrinks toward the population prior.

**Why N=16 and N=26:** the first design used rounds at N=6 and N=16. Pilot data showed accuracy at N=6 was systematically lower than at N=10 and N=14 because the first round was a warmup artifact, biasing the OLS intercept downward. The accuracy-degradation region across users turned out to sit at N=18 to N=30, but the original anchors straddled values too small to measure that degradation. Moving both anchors into the operating region (N=16 as a clean baseline, N=26 close to the typical recommended N) fixed both problems.

**Safety check:** if predicted accuracy at N* falls below 75%, the system overrides the recommendation and falls back to the N from calibration that achieved the highest measured bit rate. The 75% threshold is where the Shenoy effective accuracy factor (2p − 1) drops to 0.50, meaning errors are cutting the player's achieved bit rate to half of what it would be at the same speed with zero errors.

**Known limitation:** the calibration uses three 7-second rounds with ~15-22 trials per round. Two failure modes showed up in testing. First, players who hit 100% accuracy in calibration but degrade under scored-run pressure default to N=31 because the model has no signal for the degradation. One pilot user (~830ms mean RT) achieved perfect calibration but scored 86.1% in the 60-second run with 10 errors, producing 4.25 bps at N=31 where N=18 would have given an estimated ~4.4 bps. Second, with only ~20 trials per anchor, single-trial noise in round 2 propagates to large swings in the recommended N. The Bayesian cross-check exists specifically to flag such cases. A longer calibration or a dedicated high-pressure round could address both failure modes, at the cost of extending the familiarization period.

## Results

### Developer testing

| Session | N | Sc | Si | Accuracy | Bit Rate (bps) | Notes |
|---|---|---|---|---|---|---|
| 1 | 10 | 69 | 0 | 100% | 3.65 | First attempt, no preview |
| 2 | 10 | 114 | 4 | 96.6% | 5.81 | Lookahead added |
| 3 | 26 | 163 | 0 | 100% | 12.62 | Calibrated, all 26 letters |
| 4 | 27 | 47 | 1 | 97.9% | 14.41 | Greedy pruning active (15s) |
| 5 | 27 | 152 | 5 | 96.8% | 11.52 | Official 60s scored run |

### First-session testing

Six users tested across seven sessions, with one user repeated to measure a practice effect. Three completed the multi-round collect protocol sweeping N through {6, 10, 14, 18, 24, 30}, and the others did calibrated single runs.

| User | Mode | Best N | Best bps | Accuracy | Profile |
|---|---|---|---|---|---|
| User 4 | calibrated | 24 | 11.91 | 98.8% | Fast, high accuracy |
| User 6 (s2) | collect | 30 | 8.58 | 98.2% | Practice effect from s1 (+8%) |
| User 5 | collect | 31 | 8.59 | 100% | Slow (~570ms RT), perfect accuracy |
| User 1 | collect | 30 | 6.15 | 100% | Slow (~750ms RT), high accuracy |
| User 2 | calibrated | 24 | 5.88 | 100% | ~700ms latency |
| User 3 | calibrated | 24 | 6.94 | 96.0% | Non-monotonic accuracy |

Optimal N varied from 14 to 31, spanning nearly the full available range. No single fixed N would have been optimal for all players, validating the adaptive design. User 5 was the slowest player but made zero errors, with a fitted Hick-Hyman slope b=0.126 (the highest observed), confirming that larger alphabets genuinely slow this player while faster typists showed slopes near zero. User 3 showed a non-monotonic accuracy curve (98.2% at N=10, 86.5% at N=18, 96% at N=24), which is exactly the case the linear model p = p₀ - c·log₂(N) fails on. The N* sweep still identified N=24 correctly because it evaluates all N values empirically, not just the linear prediction.

The results screen also computes mutual information throughput I(X;Y) from the empirical confusion matrix using the Panzeri-Treves (1996) first-order bias correction, alongside the achieved bit rate and the theoretical maximum at perfect accuracy. In testing at N=28, the Shenoy rate (12.05 bps) slightly exceeded the MI throughput (11.24 bps) because Shenoy uses log₂(N-1) as a flat per-selection rate while MI accounts for the actual confusion structure, and the gap to the theoretical maximum (12.68 bps) is the information lost to errors. A speed-accuracy operating point chart shows the player's position on iso-bit-rate curves.

## Future directions

The current Bayesian fit is a two-parameter regularization on the global accuracy model. A natural extension is per-key Bayesian inference using informative priors learned from aggregate first-session data, so each key's RT and error rate would shrink toward population statistics instead of being estimated from single calibration attempts. With only ~15 trials per calibration round, OLS estimates are noisy and a hierarchical model would shrink toward the population mean. The recommended N could be reported as an interval instead of a point estimate. The calibration protocol could branch on first-round accuracy, where users above 95% would trigger a longer second round, since these are the cases where the short protocol cannot measure the accuracy decay slope. The highest-leverage move beyond the algorithm is more pilot data, since the current pool is small and a larger one would let the per-key personalization tune against population statistics instead of single-session noise.

## Development process

**Calibration debugging.** Two regressions in the calibration fitter were caught and fixed during late-stage testing. The first was a clamp on the accuracy intercept p₀ that capped it at 1.0, even though the OLS extrapolation can mathematically exceed 1.0 when both anchors hit near-perfect accuracy. The clamp shifted the fitted accuracy line downward by up to 25 percentage points, causing the N* sweep to recommend small alphabets (N=6 in one case where the data clearly supported N=24+). The second was a sign-handling bug where the OLS would compute p₀ from a negative-slope fit, then clamp the slope to zero independently, leaving p₀ at a value inconsistent with the calibration data. Both bugs were caught by comparing the model's predicted accuracy at each anchor against the directly measured accuracy in calibration, which is a structural invariant the fit should preserve. The fix made p₀ and c update together in both branches.

## References

- Shenoy, Willett, Nuyujukian & Henderson (2021). Performance Considerations for General-Purpose Typing BCIs. Stanford Technical Report #01.
- Hick, W.E. (1952). On the rate of gain of information. QJEP.
- Hyman, R. (1953). Stimulus information as a determinant of reaction time. JEP.
- Leonard, J.A. (1959). Tactual choice reactions. QJEP.
- Fitts, P.M. (1954). The information capacity of the human motor system. JEP.
- Dhakal, V., Feit, A. M., Kristensson, P. O. & Oulasvirta, A. (2018). Observations on typing from 136 million keystrokes. CHI '18.
- Panzeri, S. & Treves, A. (1996). Analytical estimates of limited sampling biases. Network.
- Soukoreff, R.W. & MacKenzie, I.S. (2004). Towards a standard for pointing device evaluation, perspectives on 27 years of Fitts' law research in HCI. Int. J. Human-Computer Studies, 61(6), 751–789.
- MacKenzie, I.S. (2015). Fitts' throughput and the remarkable case of touch-based target selection. HCI International 2015, LNCS 9170, 238–249.