# turn-1-mini: A Small Streaming System for Turn-Taking Detection

p99lab, October 2026

## Abstract

We present turn-1-mini, a causal streaming system for turn-taking in spoken English conversation. It combines a compact streaming encoder and classification heads (6.1M learned parameters) with a voice activity detector and a fixed decision policy, and uses no transcripts and no look-ahead. We evaluate it on the TurnBench development set and separate what the learned model contributes from what the policy contributes. The learned model matters for interruptions and for latency on a single channel. It detects interruptions about 570 ms earlier than the best rule (95% interval 476 to 617 ms) and with higher recall: 0.986 at a false-positive rate of 0.041, which is 1.7 points above the rule. On a single channel, as a deployed voice agent runs, it reaches an end-of-turn recall of 0.920 at a false-positive rate of 0.098 and a median latency of 938 ms; rules alone reach that recall only at about 1.5 s. With both channels available, the full system detects ends of turn with recall 0.973 at a false-positive rate of 0.080 (0.969 at 0.087 on held-out conversations). An ablation shows that this two-channel result is carried by the second channel and the decision policy; the learned model makes no measurable difference there. No TurnBench data was used to train any weight.

## 1. Introduction

A voice agent must decide when the person it is talking to has finished speaking. If it answers too early it cuts the speaker off; if it waits too long the conversation feels slow. The same system must also tell an interruption from a backchannel such as "yeah" or a laugh. Most deployed systems solve this with a silence timer, or with a large model that runs in a data centre.

This report describes turn-1-mini, a turn-taking system built around a small streaming encoder, and evaluates it on TurnBench, a benchmark of two-channel English conversations with end-of-turn and interruption labels. We ask a specific question: what does a small learned model add over voice activity and rules? Our interest is in models small enough to run on the user's device, although we have not yet measured speed on one. The findings are:

1. On interruptions the learned model commits about 570 ms earlier than rules and adds 1.7 points of recall (342 against 336 of 347 interruptions).
2. On a single channel its benefit for end of turn is latency. It reaches a recall of 0.920 at a median latency under 1 s, which no rule set we tried reaches within the false-positive budget; rules alone need about 1.5 s for the same recall.
3. With two channels, a decision policy over voice activity alone reaches the same end-of-turn recall as the full system. We report this ablation in full.
4. The system is strictly causal: a streaming encoder with cached state, three small heads and a fixed policy, verified with a truncation test.

## 2. Model

Table 1 lists the components. The front-end, the encoder and the heads are causal.

**Table 1.** Components of turn-1-mini.

| Component | Function | Parameters | Size |
| --- | --- | --- | --- |
| Voice activity detector (Silero VAD v5, MIT) | Speech probability per 32 ms, per channel | third-party | 2.3 MB |
| Streaming encoder | Two convolutions and six transformer layers of width 256 with a 1.2 s attention window; one output per 20 ms; run in 160 ms steps with cached state | 4,997,632 | 20 MB float32 |
| Turn head | P(the speaker's turn is over), from the speaker's own channel | 264,705 | 1 MB |
| Cross-channel head | Reads the encoder outputs of both channels and 16 voice-activity features. Predicts whether the current silence is an end of turn, and whether the other speaker's new vocalisation claims the floor. Used for end of turn | 417,990 | 1.7 MB |
| Own-channel head | The same architecture with the speaker's own channel as its only input. Predicts the class of a vocalisation that has just started: turn, floor-taking interruption, non-floor-taking interruption, backchannel or non-content. Used for interruptions and for single-channel operation | 417,990 | 1.7 MB |
| Decision policy | Fixed rules with 18 settings chosen on the development set (Section 3) | 0 | |

The learned components total 6,098,317 parameters; with the voice activity detector's 16 kHz branch (309,633) the system has 6,407,950. For two channels the encoder runs once per channel. Not every head is used in every mode. The two-channel system uses the encoder, the cross-channel head (end of turn) and the own-channel head (interruptions): 5,833,612 parameters. Single-channel operation uses the encoder, the turn head and the own-channel head: 5,680,327 parameters.

**Front-end.** Each channel is low-pass filtered and decimated from 48 kHz to 16 kHz with a causal filter. A causal automatic gain control scales each 160 ms step by a gain computed from the audio before that step. The voice activity detector runs on this signal. The encoder receives the same signal band-limited to 4 kHz, the band it was trained on.

## 3. Decision policy

The benchmark scores committed events, not probabilities. The policy turns the model's scores into events. The settings given below are those of the 0.08 operating point (Section 5.1); the 0.10 operating point reported alongside it uses different constants.

**End of turn.** A candidate opens whenever speech stops on a speaker's channel, at time `t`. An event is committed at the earliest of the following, and never earlier than 0.6 s after `t`:

1. the cross-channel head's end-of-turn score reaches 0.97 (the score is read from 0.16 s after speech stopped);
2. the other speaker has been vocalising for 0.3 s;
3. the cross-channel head classifies the other speaker's new vocalisation as claiming the floor (score of 0.54 or more, read 0.32 to 1.2 s after its onset);
4. 2.5 s of silence have passed.

A candidate is cancelled if the speaker resumes within 0.2 s, or resumes later and continues for 0.5 s. A short laugh or "yeah" therefore does not cancel it. Three further rules apply:

- **Floor held by the other speaker.** If the other speaker had been vocalising for 1.5 s when this speaker stopped and is still vocalising 0.6 s later, the event is committed even if this speaker is heard again. This follows the benchmark's definition, in which a segment that ends while the other speaker is mid-turn is an end of turn.
- **Confirmation.** If the candidate is still open 2.5 s after `t`, a second event is committed. About 7% of end-of-turn labels on the development set lie one to three seconds after the last detected speech on that channel; the confirmation event covers these. At most one confirmation is emitted per candidate. At this operating point the silence deadline is also 2.5 s, and events with the same timestamp are one event: a candidate that reaches 2.5 s with no earlier event gets a single event, and the confirmation is a second event only when the first was committed earlier by another trigger.
- **End of stream.** If the audio ends while a candidate is open, an event is committed at the final sample.

**Interruption.** At most one event is committed per vocalisation: when the own-channel head's floor-claim score reaches 0.77 between 0.16 and 1.2 s after the onset, and otherwise when the vocalisation has lasted 1.2 s.

## 4. Training

No TurnBench audio or annotation, from the development or the test set, was used for training, calibration or model selection. The benchmark's training set was not used.

The system is built on open speech encoders and trained on licensed conversational speech. The training data carries no turn-taking labels; we derived them from each speaker's voice activity and word timings, following the benchmark's published definitions.

## 5. Experiments

### 5.1 Setup

We evaluate on the TurnBench development set (38 English conversations, 7.3 h) with the benchmark's scorer. For end of turn it contains 1,904 annotated ends of turn and 1,063 pauses and backchannels on which a false positive can be counted; for interruption, 347 interruptions and 3,733 other vocalisations. Test labels are not public, so every number below is a development-set number.

**Operating point.** Following the benchmark's protocol, an operating point is the setting with the highest recall whose development false-positive rate stays within a budget. We report budgets of 0.08 and 0.10 for end of turn and take 0.08 as the system's operating point (Section 5.4); for interruption the budget is 0.05. The policy has 18 settings chosen on the development set: 14 numeric thresholds and durations (10 for end of turn, 4 for interruption) and 4 discrete choices. They were selected by scoring several thousand sampled settings per policy family. Eleven further constants, such as the voice activity threshold, the 0.2 s resumption gap and the 1.2 s read window, were fixed in advance.

**Held-out estimate.** Because the constants are selected and scored on the same 38 conversations, development scores are optimistic. We therefore also report a split-half estimate: the constants are selected on 19 random conversations and scored on the other 19, repeated 1,000 times.

**Latency.** Latency is the benchmark's: the committed time minus the annotated time of the event. The policy's waits are counted from the detected end of speech, which precedes the annotated end of turn by a median of 90 ms. A median latency can therefore be shorter than the policy's 0.6 s minimum wait.

**Hardware.** All experiments ran on a GPU and measure accuracy. They say nothing about speed on a phone or laptop processor, which we have not measured.

### 5.2 What the learned model contributes

The learned model matters where a second channel cannot do the work: on interruptions, and when only one channel is available. Table 2 compares it with rules that use voice activity only.

**Table 2.** Learned model against rules (recall / false-positive rate / median latency).

| Setting | Budget | Rules only | With the learned model |
| --- | --- | --- | --- |
| Interruption, two channels | 0.05 | 0.968 / 0.050 / 1134 ms | 0.986 / 0.041 / 562 ms |
| Interruption, one channel | 0.05 | 0.963 / 0.048 / 1026 ms | 0.986 / 0.041 / 562 ms |
| End of turn, one channel, silence deadline only | 0.08 | 0.814 / 0.077 / 1409 ms | 0.857 / 0.073 / 997 ms |
| End of turn, one channel, silence deadline only | 0.10 | 0.850 / 0.094 / 1249 ms | 0.872 / 0.100 / 995 ms |
| End of turn, one channel, with a cancellation rule and end of stream, no latency limit | 0.10 | 0.935 / 0.096 / 1464 ms | 0.927 / 0.099 / 1358 ms |
| The same, median latency held at or under 1 s | 0.10 | not reachable within the budget | 0.920 / 0.098 / 938 ms |

**Interruptions.** The model's median latency is 572 ms shorter than that of two-channel rules (paired bootstrap over the 38 conversations, 95% interval 476 to 617 ms). It also adds 1.7 points of recall (342 against 336 of the 347 interruptions; interval 0.4 to 3.3 points), at a false-positive rate that is no higher. Against one-channel rules it is 464 ms earlier (366 to 512 ms) and 2.3 points higher (0.9 to 4.1 points). The interruption decision uses the speaker's own channel only, so the system's result is the same with one channel as with two.

**End of turn on one channel.** Against a plain silence timer the model gives more recall: 4.3 points at the 0.08 budget (Table 2), or 3.5 to 4.6 points depending on how the timer's false-positive rate is matched to the model's, and 1.7 to 2.0 points when the two are compared at the same false-positive rate near 0.10. Against the best rules the picture is different. A timer with a cancellation rule (a resumption must last before it cancels) reaches 0.935 without any model, slightly above the model with the same rules, but only at a median latency of 1.5 s. When the median latency is held under 1 s, no rule set stays within the false-positive budget (the best reaches 0.895 at a false-positive rate of 0.150), while the model reaches 0.920 at 0.098. On one channel the model therefore buys about half a second of latency, not recall. The single-channel rows describe a deployed agent, which hears one human speaker and commits one event per turn.

**Which head for interruptions.** We also tried the cross-channel head for this decision. It gives 0.986 / 0.049 / 688 ms: the same recall, more false positives and a later decision. It divides its probability between floor-taking and non-floor-taking interruptions, which differ only in what happens afterwards, and so it separates interruptions from backchannels less sharply at onset than the own-channel head does (AUC 0.77 against 0.92 at 0.2 s on held-out conversations; none of these conversations was used in training, and 113 of their turns had been used for earlier model selection). The system therefore uses the own-channel head for interruptions.

### 5.3 Two-channel end of turn

**Table 3.** Full system, end of turn, two channels.

| Budget | Protocol | Recall | False-positive rate | Median latency |
| --- | --- | --- | --- | --- |
| 0.08 | Selected and scored on the development set | 0.973 | 0.080 | 603 ms |
| 0.08 | Split-half, held-out half | 0.969 | 0.087 | |
| 0.10 | Selected and scored on the development set | 0.977 | 0.099 | 568 ms |
| 0.10 | Split-half, held-out half | 0.973 | 0.106 | |

By conversation type, development recall ranges from 0.93 to 0.99 (lowest on Narrative) and the false-positive rate from 0.05 to 0.12 (highest on Casual and Argumentative).

Table 4 removes one component at a time; each row is re-tuned to the same false-positive budget.

**Table 4.** Ablation, end of turn, two channels (recall / false-positive rate / median latency).

| System | Budget 0.08 | Budget 0.10 |
| --- | --- | --- |
| Full system | 0.973 / 0.080 / 603 ms | 0.977 / 0.099 / 568 ms |
| Without learned components (voice activity and policy only) | 0.973 / 0.079 / 602 ms | 0.976 / 0.090 / 599 ms |
| Without the confirmation event | 0.935 / 0.076 / 648 ms | 0.941 / 0.099 / 1056 ms |

The second channel and the decision policy carry this result. The learned components make no measurable difference: in a paired bootstrap over the 38 conversations the recall difference between the full system and the rules-only system is +0.0005 at the 0.08 budget (95% interval -0.0030 to +0.0038) and +0.0016 at the 0.10 budget (-0.0049 to +0.0065). The confirmation event adds 3.6 to 3.8 points. We keep the cross-channel head in the system as evaluated, although this ablation gives no evidence that it helps.

Without the confirmation event the median latency at the 0.10 budget rises to 1056 ms. This is an effect of re-tuning: with a single event per turn, the best setting at that budget waits until the other speaker has been vocalising for 1.0 s instead of 0.3 s, so that its one event still falls inside the scoring window of labels that are placed late. At the 0.08 budget the selected setting is a different one and the latency is 648 ms.

For reference, one published system has development-set predictions: VAP scores 0.841 / 0.045 / 463 ms on end of turn and 0.957 / 0.100 / 896 ms on interruption at its own operating point. The leading entries publish test results only, which are not comparable with development numbers.

### 5.4 Robustness of the operating point

A submission is valid only if its test false-positive rate stays at or below 0.15, so the margin matters. Table 5 gives two estimates of the false-positive rate on unseen conversations.

**Table 5.** End-of-turn false-positive rate on unseen conversations.

| Budget | Split-half, held-out: mean / 95th percentile / halves above 0.15 | Resampled 116-conversation set: mean / 99th percentile |
| --- | --- | --- |
| 0.08 | 0.087 / 0.133 / 1.1% | 0.080 / 0.108 |
| 0.10 | 0.106 / 0.156 / 6.7% | 0.099 / 0.129 |

Neither estimate is exact. A half of 19 conversations varies more than a test set of 116 would, so the split-half spread is too wide. The resampled set is drawn from the same conversations the constants were tuned on, so it inherits their optimism and is too narrow. The published baselines give a third reference: between development and test their false-positive rates moved by -0.022 on average, and by +0.031 in the worst case.

At the 0.10 budget the split-half 95th percentile is above the limit. We therefore take the 0.08 budget as the operating point. It costs 0.4 points of recall and leaves a wide margin.

## 6. Causality

Every committed timestamp is the time at which the last audio used by that decision had been heard.

- Model outputs are read at the end of each 160 ms step, and that step end is the timestamp.
- Timer and other-channel conditions are stamped at the 32 ms voice-activity chunk boundary at which they became true.
- "Still vocalising" means that the last speech chunk ended at most 0.2 s earlier, as known at that moment.
- The gain applied to a step is computed from audio before the step. The resampling and band-limit filters are causal. No output is smoothed or re-aligned afterwards.

We verified this with a truncation test. Three development conversations were cut at 200 s and the full pipeline was run again on the truncated audio. All model outputs available by 200 s were identical to those of the full run (maximum difference 0), as were all events committed before 200 s (204 end-of-turn and 158 interruption events). The end-of-stream event is excluded from this comparison by construction.

## 7. Conclusion

turn-1-mini is a causal turn-taking system with 6.1M learned parameters, trained without benchmark data. Its learned model earns its place on interruptions, where it is about 570 ms earlier and 1.7 points more accurate than rules, and on a single channel, where it reaches a given end-of-turn recall about half a second sooner than rules can. On two-channel recordings the end-of-turn result belongs to the decision policy and the second channel.

Five limitations remain. The policy constants were tuned on the 38 development conversations, so development scores are optimistic; the split-half estimate is the fairer number. The heads were trained on labels that we derived by rule, and their outputs feed a rule-based policy, so part of what they learn may be the rules themselves; human-labelled training data would test this. The benchmark's channels are clean, whereas a deployed microphone also picks up the agent's own voice. All experiments ran on a GPU and measure accuracy; speed on a phone or laptop processor, which the on-device aim depends on, has not been measured yet. The system has been trained and evaluated on English only.

## Contact

p99lab, https://p99lab.com
