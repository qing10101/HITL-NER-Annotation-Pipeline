# LLM reconstruction attack: recon_claude-fable-5-1_subagent_s0

Unit = masked span (501 in every condition). exact@1/@5 = normalised top-1 / any-of-5 guess equals the removed text; cat@1 = top guess in the span type's lexicon; anycat@1 = in any privacy lexicon; conf = self-reported confidence in the top guess (not comparable to T5's teacher-forced gold p).

| attacker | cond | label | n | exact@1 | exact@5 | cat@1 | anycat@1 | conf / gold p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context-free prior | any | all | 501 | 0.098 | 0.299 | 0.413 | – | – |
| context-free prior | any | MINOR_AGE | 165 | 0.000 | 0.145 | 0.000 | – | – |
| context-free prior | any | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior | any | GEN_NOUN | 97 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior | any | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior | any | FAM_KIN | 207 | 0.237 | 0.609 | 1.000 | – | – |
| type-oracle prior | any | all | 501 | 0.236 | 0.585 | 1.000 | – | – |
| type-oracle prior | any | MINOR_AGE | 165 | 0.145 | 0.424 | 1.000 | – | – |
| type-oracle prior | any | MINOR_EDU | 10 | 0.000 | 0.200 | 1.000 | – | – |
| type-oracle prior | any | GEN_NOUN | 97 | 0.443 | 0.753 | 1.000 | – | – |
| type-oracle prior | any | GEN_PHYS | 22 | 0.091 | 0.227 | 1.000 | – | – |
| type-oracle prior | any | FAM_KIN | 207 | 0.237 | 0.691 | 1.000 | – | – |
| t5-base_adaptive | gold | all | 501 | 0.261 | 0.471 | 0.669 | 0.924 | 0.235 |
| t5-base_adaptive | gold | MINOR_AGE | 165 | 0.158 | 0.309 | 0.600 | 0.915 | 0.201 |
| t5-base_adaptive | gold | MINOR_EDU | 10 | 0.100 | 0.200 | 0.400 | 0.700 | 0.096 |
| t5-base_adaptive | gold | GEN_NOUN | 97 | 0.216 | 0.526 | 0.433 | 0.959 | 0.199 |
| t5-base_adaptive | gold | GEN_PHYS | 22 | 0.091 | 0.273 | 0.318 | 0.409 | 0.158 |
| t5-base_adaptive | gold | FAM_KIN | 207 | 0.391 | 0.609 | 0.884 | 0.981 | 0.295 |
| t5-base_adaptive | loo | all | 501 | 0.283 | 0.575 | 0.679 | 0.916 | 0.250 |
| t5-base_adaptive | loo | MINOR_AGE | 165 | 0.139 | 0.352 | 0.552 | 0.891 | 0.199 |
| t5-base_adaptive | loo | MINOR_EDU | 10 | 0.100 | 0.200 | 0.400 | 0.700 | 0.161 |
| t5-base_adaptive | loo | GEN_NOUN | 97 | 0.247 | 0.608 | 0.505 | 0.948 | 0.215 |
| t5-base_adaptive | loo | GEN_PHYS | 22 | 0.091 | 0.273 | 0.409 | 0.455 | 0.155 |
| t5-base_adaptive | loo | FAM_KIN | 207 | 0.444 | 0.787 | 0.903 | 0.981 | 0.321 |
| t5-base_adaptive | rand | all | 501 | 0.222 | 0.331 | – | 0.130 | 0.196 |
| t5-base_adaptive | rand | RANDOM | 501 | 0.222 | 0.331 | – | 0.130 | 0.196 |
| t5-base_nonadaptive | gold | all | 501 | 0.078 | 0.172 | 0.216 | 0.301 | 0.102 |
| t5-base_nonadaptive | gold | MINOR_AGE | 165 | 0.006 | 0.048 | 0.121 | 0.248 | 0.080 |
| t5-base_nonadaptive | gold | MINOR_EDU | 10 | 0.000 | 0.100 | 0.000 | 0.000 | 0.026 |
| t5-base_nonadaptive | gold | GEN_NOUN | 97 | 0.062 | 0.258 | 0.196 | 0.309 | 0.084 |
| t5-base_nonadaptive | gold | GEN_PHYS | 22 | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 |
| t5-base_nonadaptive | gold | FAM_KIN | 207 | 0.150 | 0.246 | 0.329 | 0.382 | 0.138 |
| t5-base_nonadaptive | loo | all | 501 | 0.170 | 0.335 | 0.401 | 0.485 | 0.164 |
| t5-base_nonadaptive | loo | MINOR_AGE | 165 | 0.079 | 0.188 | 0.352 | 0.479 | 0.127 |
| t5-base_nonadaptive | loo | MINOR_EDU | 10 | 0.000 | 0.100 | 0.100 | 0.100 | 0.040 |
| t5-base_nonadaptive | loo | GEN_NOUN | 97 | 0.113 | 0.361 | 0.278 | 0.412 | 0.115 |
| t5-base_nonadaptive | loo | GEN_PHYS | 22 | 0.045 | 0.045 | 0.091 | 0.091 | 0.046 |
| t5-base_nonadaptive | loo | FAM_KIN | 207 | 0.290 | 0.483 | 0.546 | 0.585 | 0.235 |
| t5-base_nonadaptive | rand | all | 501 | 0.230 | 0.353 | – | 0.018 | 0.224 |
| t5-base_nonadaptive | rand | RANDOM | 501 | 0.230 | 0.353 | – | 0.018 | 0.224 |
| claude-fable-5-1_subagent | gold | all | 501 | 0.361 | 0.651 | 0.675 | 0.848 | 0.355 |
| claude-fable-5-1_subagent | gold | MINOR_AGE | 165 | 0.230 | 0.485 | 0.564 | 0.855 | 0.301 |
| claude-fable-5-1_subagent | gold | MINOR_EDU | 10 | 0.200 | 0.400 | 0.600 | 0.600 | 0.370 |
| claude-fable-5-1_subagent | gold | GEN_NOUN | 97 | 0.464 | 0.835 | 0.691 | 0.907 | 0.399 |
| claude-fable-5-1_subagent | gold | GEN_PHYS | 22 | 0.273 | 0.409 | 0.318 | 0.318 | 0.344 |
| claude-fable-5-1_subagent | gold | FAM_KIN | 207 | 0.435 | 0.734 | 0.797 | 0.884 | 0.377 |
| claude-fable-5-1_subagent | loo | all | 501 | 0.439 | 0.717 | 0.727 | 0.864 | 0.396 |
| claude-fable-5-1_subagent | loo | MINOR_AGE | 165 | 0.255 | 0.588 | 0.630 | 0.848 | 0.326 |
| claude-fable-5-1_subagent | loo | MINOR_EDU | 10 | 0.200 | 0.300 | 0.600 | 0.600 | 0.440 |
| claude-fable-5-1_subagent | loo | GEN_NOUN | 97 | 0.515 | 0.845 | 0.732 | 0.918 | 0.426 |
| claude-fable-5-1_subagent | loo | GEN_PHYS | 22 | 0.273 | 0.409 | 0.227 | 0.227 | 0.355 |
| claude-fable-5-1_subagent | loo | FAM_KIN | 207 | 0.580 | 0.812 | 0.860 | 0.932 | 0.440 |
| claude-fable-5-1_subagent | rand | all | 501 | 0.451 | 0.673 | – | 0.030 | 0.458 |
| claude-fable-5-1_subagent | rand | RANDOM | 501 | 0.451 | 0.673 | – | 0.030 | 0.458 |
| claude-fable-5-1_subagent | rand_s43 | all | 501 | 0.447 | 0.699 | – | 0.044 | 0.442 |
| claude-fable-5-1_subagent | rand_s43 | RANDOM | 501 | 0.447 | 0.699 | – | 0.044 | 0.442 |
| claude-fable-5-1_subagent | rand_s44 | all | 501 | 0.427 | 0.671 | – | 0.026 | 0.445 |
| claude-fable-5-1_subagent | rand_s44 | RANDOM | 501 | 0.427 | 0.671 | – | 0.026 | 0.445 |

## Paired contrasts (row-level bootstrap 95% CI, exact McNemar on exact@1)

| contrast | n | metric | A | B | A−B | CI95 | McNemar b/c | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gold_vs_loo_multispan | 320 | exact1 | 0.369 | 0.491 | -0.122 | [-0.177, -0.061] | 14/53 | 0.000 |
| gold_vs_loo_multispan | 320 | exact5 | 0.650 | 0.753 | -0.103 | [-0.156, -0.052] | 12/45 | 0.000 |
| gold_vs_loo_multispan | 320 | conf1 | 0.342 | 0.406 | -0.065 | [-0.083, -0.047] | – | – |
| gold_vs_loo_all | 501 | exact1 | 0.361 | 0.439 | -0.078 | [-0.117, -0.040] | 14/53 | 0.000 |
| gold_vs_loo_all | 501 | exact5 | 0.651 | 0.717 | -0.066 | [-0.100, -0.034] | 12/45 | 0.000 |
| gold_vs_loo_all | 501 | conf1 | 0.355 | 0.396 | -0.041 | [-0.054, -0.030] | – | – |
| gold_vs_rand | 501 | exact1 | 0.361 | 0.451 | -0.090 | [-0.149, -0.029] | 86/131 | 0.003 |
| gold_vs_rand | 501 | exact5 | 0.651 | 0.673 | -0.022 | [-0.079, +0.038] | 100/111 | 0.491 |
| gold_vs_rand | 501 | conf1 | 0.355 | 0.458 | -0.103 | [-0.124, -0.080] | – | – |
| gold_vs_rand_s43 | 501 | exact1 | 0.361 | 0.447 | -0.086 | [-0.142, -0.027] | 85/128 | 0.004 |
| gold_vs_rand_s43 | 501 | exact5 | 0.651 | 0.699 | -0.048 | [-0.104, +0.007] | 86/110 | 0.100 |
| gold_vs_rand_s43 | 501 | conf1 | 0.355 | 0.442 | -0.088 | [-0.107, -0.068] | – | – |
| gold_vs_rand_s44 | 501 | exact1 | 0.361 | 0.427 | -0.066 | [-0.120, -0.010] | 89/122 | 0.027 |
| gold_vs_rand_s44 | 501 | exact5 | 0.651 | 0.671 | -0.020 | [-0.076, +0.037] | 93/103 | 0.520 |
| gold_vs_rand_s44 | 501 | conf1 | 0.355 | 0.445 | -0.090 | [-0.110, -0.071] | – | – |
