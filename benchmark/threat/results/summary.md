# Threat experiment results

Runs found: 4 presence, 5 recon (seeds averaged where repeated).

## Scheme B: dimension-presence attacker

AUROC per label, macro AUROC, mean probability on positives (p+), attack success (p>=0.8 and positive). Train-on `raw` = non-adaptive, `gold_del` = adaptive.

| attacker | test cond | minor AUROC | gender AUROC | kinship AUROC | macro AUROC | macro F1 | p+ minor | p+ gender | p+ kinship | succ minor | succ gender | succ kinship |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neobert_gold_del | raw | 0.704 | 0.784 | 0.773 | 0.754 | 0.535 | 0.638 | 0.479 | 0.754 | 0.493 | 0.264 | 0.712 |
| neobert_gold_del | gold_del | 0.743 | 0.743 | 0.791 | 0.759 | 0.580 | 0.715 | 0.487 | 0.802 | 0.681 | 0.245 | 0.782 |
| neobert_gold_del | rand_del | 0.687 | 0.779 | 0.750 | 0.738 | 0.521 | 0.598 | 0.469 | 0.718 | 0.471 | 0.226 | 0.659 |
| neobert_raw | raw | 0.920 | 0.944 | 0.957 | 0.940 | 0.826 | 0.819 | 0.682 | 0.921 | 0.797 | 0.613 | 0.894 |
| neobert_raw | gold_del | 0.695 | 0.738 | 0.723 | 0.719 | 0.317 | 0.297 | 0.154 | 0.362 | 0.203 | 0.094 | 0.300 |
| neobert_raw | rand_del | 0.920 | 0.939 | 0.958 | 0.939 | 0.825 | 0.825 | 0.680 | 0.915 | 0.804 | 0.623 | 0.876 |
| tfidf_gold_del | raw | 0.757 | 0.708 | 0.804 | 0.756 | 0.436 | 0.488 | 0.209 | 0.581 | 0.217 | 0.028 | 0.312 |
| tfidf_gold_del | gold_del | 0.773 | 0.740 | 0.832 | 0.782 | 0.544 | 0.569 | 0.264 | 0.713 | 0.304 | 0.028 | 0.512 |
| tfidf_gold_del | rand_del | 0.738 | 0.707 | 0.771 | 0.739 | 0.411 | 0.454 | 0.199 | 0.525 | 0.196 | 0.019 | 0.288 |
| tfidf_raw | raw | 0.942 | 0.955 | 0.983 | 0.960 | 0.850 | 0.754 | 0.690 | 0.855 | 0.601 | 0.566 | 0.735 |
| tfidf_raw | gold_del | 0.754 | 0.727 | 0.786 | 0.755 | 0.250 | 0.316 | 0.139 | 0.343 | 0.072 | 0.019 | 0.059 |
| tfidf_raw | rand_del | 0.943 | 0.958 | 0.982 | 0.961 | 0.847 | 0.751 | 0.685 | 0.855 | 0.609 | 0.557 | 0.753 |

**Top positive word features, presence_tfidf_gold_del_s42** (logistic-regression coefficients)

- minor: my (10.262), loves (5.762), our (5.596), she (4.685), my is (4.683), my and (4.46), my can (4.069), his (4.042), my who (4.017), the love (3.764), what the (3.742), my in (3.502), we (3.497), it all (3.27), for my (3.091), her (2.946), the to (2.944), and love (2.917), the and (2.902), for she (2.822)
- gender: my (10.191), my and (7.104), he (5.058), my has (4.579), we (4.421), bra (4.138), am and (3.71), requested (3.603), belt (3.431), and my (3.217), is happy (3.104), me (3.084), likes this (3.014), on him (3.006), small for (3.004), smells great (3.0), and loves (2.987), happy would (2.984), nice quality (2.937), my to (2.881)
- kinship: my (17.41), my love (8.87), my is (7.818), my and (7.725), my loved (7.494), my who (5.443), my has (5.408), for my (5.28), our (5.264), her (5.044), loves (4.817), mi (4.811), my for (4.715), my to (4.455), but my (4.453), he (3.863), my are (3.711), she (3.459), the love (3.412), who is (3.208)

**Top positive word features, presence_tfidf_raw_s42** (logistic-regression coefficients)

- minor: old (11.492), baby (8.579), my baby (8.474), year old (8.453), year (7.35), little one (6.714), my (6.63), my year (6.251), my toddler (6.208), the baby (5.146), my little (4.954), yr (4.858), she (4.705), for my (4.468), yr old (4.35), month old (4.333), age (3.951), lo (3.946), toddler (3.905), our (3.772)
- gender: wife (12.027), my wife (10.908), my husband (10.423), husband (10.3), hubby (6.487), my (5.515), woman (4.338), beard (3.975), mom (3.967), bra (3.951), my boyfriend (3.61), boyfriend (3.404), me (3.178), and my (3.163), girlfriend (3.017), my beard (2.973), he (2.769), pumping (2.722), am (2.565), in (2.395)
- kinship: son (16.124), my kids (12.723), my (12.126), my son (11.554), daughter (9.052), my daughter (7.922), my mom (6.735), my child (6.236), my kid (5.742), for my (5.563), mom (5.508), niece (5.375), grandson (5.272), mi (4.665), kid (4.212), my mother (4.083), sister (3.994), my girls (3.962), he (3.916), mother (3.707)

## Scheme A: span reconstruction attacker

exact@1/@5 = normalised prediction equals the masked span; cat@1 = top prediction falls in the same entity type's lexicon; anycat@1 = in any privacy lexicon; gold p = teacher-forced probability of the gold span. `nonadaptive` never saw a privacy span (trained on random spans of negative rows).

Caveat: the `t5gemma2_*_fp32` attackers learned little beyond the training-span prior (context swap changes gold-span probability by a log-ratio of 0.2 vs 3.3 for t5-base; see docs/threat_experiment_design.md, section 8); `t5-base_*_fp32` is the usable reconstruction attacker. Earlier bf16 runs are invalid and kept in results/invalid_bf16/.

| attacker | cond | label | n | exact@1 | exact@5 | cat@1 | anycat@1 | gold p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-fable-5-1_subagent | gold | all | 501 | 0.361 | 0.651 | 0.675 | 0.848 | – |
| claude-fable-5-1_subagent | gold | FAM_KIN | 207 | 0.435 | 0.734 | 0.797 | 0.884 | – |
| claude-fable-5-1_subagent | gold | MINOR_AGE | 165 | 0.230 | 0.485 | 0.564 | 0.855 | – |
| claude-fable-5-1_subagent | gold | MINOR_EDU | 10 | 0.200 | 0.400 | 0.600 | 0.600 | – |
| claude-fable-5-1_subagent | gold | GEN_NOUN | 97 | 0.464 | 0.835 | 0.691 | 0.907 | – |
| claude-fable-5-1_subagent | gold | GEN_PHYS | 22 | 0.273 | 0.409 | 0.318 | 0.318 | – |
| claude-fable-5-1_subagent | loo | all | 501 | 0.439 | 0.717 | 0.727 | 0.864 | – |
| claude-fable-5-1_subagent | loo | FAM_KIN | 207 | 0.580 | 0.812 | 0.860 | 0.932 | – |
| claude-fable-5-1_subagent | loo | MINOR_AGE | 165 | 0.255 | 0.588 | 0.630 | 0.848 | – |
| claude-fable-5-1_subagent | loo | MINOR_EDU | 10 | 0.200 | 0.300 | 0.600 | 0.600 | – |
| claude-fable-5-1_subagent | loo | GEN_NOUN | 97 | 0.515 | 0.845 | 0.732 | 0.918 | – |
| claude-fable-5-1_subagent | loo | GEN_PHYS | 22 | 0.273 | 0.409 | 0.227 | 0.227 | – |
| claude-fable-5-1_subagent | rand | all | 501 | 0.451 | 0.673 | – | 0.030 | – |
| claude-fable-5-1_subagent | rand | RANDOM | 501 | 0.451 | 0.673 | – | 0.030 | – |
| t5-base_adaptive_fp32 | gold | all | 501 | 0.261 | 0.471 | 0.669 | 0.924 | 0.235 |
| t5-base_adaptive_fp32 | gold | FAM_KIN | 207 | 0.391 | 0.609 | 0.884 | 0.981 | 0.295 |
| t5-base_adaptive_fp32 | gold | MINOR_AGE | 165 | 0.158 | 0.309 | 0.600 | 0.915 | 0.201 |
| t5-base_adaptive_fp32 | gold | MINOR_EDU | 10 | 0.100 | 0.200 | 0.400 | 0.700 | 0.096 |
| t5-base_adaptive_fp32 | gold | GEN_NOUN | 97 | 0.216 | 0.526 | 0.433 | 0.959 | 0.199 |
| t5-base_adaptive_fp32 | gold | GEN_PHYS | 22 | 0.091 | 0.273 | 0.318 | 0.409 | 0.158 |
| t5-base_adaptive_fp32 | loo | all | 501 | 0.283 | 0.575 | 0.679 | 0.916 | 0.250 |
| t5-base_adaptive_fp32 | loo | FAM_KIN | 207 | 0.444 | 0.787 | 0.903 | 0.981 | 0.321 |
| t5-base_adaptive_fp32 | loo | MINOR_AGE | 165 | 0.139 | 0.352 | 0.552 | 0.891 | 0.199 |
| t5-base_adaptive_fp32 | loo | MINOR_EDU | 10 | 0.100 | 0.200 | 0.400 | 0.700 | 0.161 |
| t5-base_adaptive_fp32 | loo | GEN_NOUN | 97 | 0.247 | 0.608 | 0.505 | 0.948 | 0.215 |
| t5-base_adaptive_fp32 | loo | GEN_PHYS | 22 | 0.091 | 0.273 | 0.409 | 0.455 | 0.155 |
| t5-base_adaptive_fp32 | rand | all | 501 | 0.222 | 0.331 | – | 0.130 | 0.196 |
| t5-base_adaptive_fp32 | rand | RANDOM | 501 | 0.222 | 0.331 | – | 0.130 | 0.196 |
| t5-base_nonadaptive_fp32 | gold | all | 501 | 0.078 | 0.172 | 0.216 | 0.301 | 0.102 |
| t5-base_nonadaptive_fp32 | gold | FAM_KIN | 207 | 0.150 | 0.246 | 0.329 | 0.382 | 0.138 |
| t5-base_nonadaptive_fp32 | gold | MINOR_AGE | 165 | 0.006 | 0.048 | 0.121 | 0.248 | 0.080 |
| t5-base_nonadaptive_fp32 | gold | MINOR_EDU | 10 | 0.000 | 0.100 | 0.000 | 0.000 | 0.026 |
| t5-base_nonadaptive_fp32 | gold | GEN_NOUN | 97 | 0.062 | 0.258 | 0.196 | 0.309 | 0.084 |
| t5-base_nonadaptive_fp32 | gold | GEN_PHYS | 22 | 0.045 | 0.045 | 0.045 | 0.045 | 0.045 |
| t5-base_nonadaptive_fp32 | loo | all | 501 | 0.170 | 0.335 | 0.401 | 0.485 | 0.164 |
| t5-base_nonadaptive_fp32 | loo | FAM_KIN | 207 | 0.290 | 0.483 | 0.546 | 0.585 | 0.235 |
| t5-base_nonadaptive_fp32 | loo | MINOR_AGE | 165 | 0.079 | 0.188 | 0.352 | 0.479 | 0.127 |
| t5-base_nonadaptive_fp32 | loo | MINOR_EDU | 10 | 0.000 | 0.100 | 0.100 | 0.100 | 0.040 |
| t5-base_nonadaptive_fp32 | loo | GEN_NOUN | 97 | 0.113 | 0.361 | 0.278 | 0.412 | 0.115 |
| t5-base_nonadaptive_fp32 | loo | GEN_PHYS | 22 | 0.045 | 0.045 | 0.091 | 0.091 | 0.046 |
| t5-base_nonadaptive_fp32 | rand | all | 501 | 0.230 | 0.353 | – | 0.018 | 0.224 |
| t5-base_nonadaptive_fp32 | rand | RANDOM | 501 | 0.230 | 0.353 | – | 0.018 | 0.224 |
| t5gemma2_adaptive_fp32 | gold | all | 501 | 0.108 | 0.307 | 0.497 | 0.942 | 0.209 |
| t5gemma2_adaptive_fp32 | gold | FAM_KIN | 207 | 0.242 | 0.507 | 0.826 | 0.918 | 0.249 |
| t5gemma2_adaptive_fp32 | gold | MINOR_AGE | 166 | 0.018 | 0.133 | 0.458 | 0.964 | 0.223 |
| t5gemma2_adaptive_fp32 | gold | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | 0.800 | 0.033 |
| t5gemma2_adaptive_fp32 | gold | GEN_NOUN | 96 | 0.010 | 0.281 | 0.021 | 0.969 | 0.156 |
| t5gemma2_adaptive_fp32 | gold | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | 0.955 | 0.046 |
| t5gemma2_adaptive_fp32 | loo | all | 501 | 0.120 | 0.375 | 0.469 | 1.000 | 0.202 |
| t5gemma2_adaptive_fp32 | loo | FAM_KIN | 207 | 0.271 | 0.628 | 0.894 | 1.000 | 0.265 |
| t5gemma2_adaptive_fp32 | loo | MINOR_AGE | 166 | 0.012 | 0.114 | 0.277 | 1.000 | 0.171 |
| t5gemma2_adaptive_fp32 | loo | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | 1.000 | 0.028 |
| t5gemma2_adaptive_fp32 | loo | GEN_NOUN | 96 | 0.021 | 0.406 | 0.042 | 1.000 | 0.174 |
| t5gemma2_adaptive_fp32 | loo | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | 1.000 | 0.048 |
| t5gemma2_adaptive_fp32 | rand | all | 501 | 0.000 | 0.000 | – | 0.946 | 0.015 |
| t5gemma2_adaptive_fp32 | rand | RANDOM | 501 | 0.000 | 0.000 | – | 0.946 | 0.015 |
| t5gemma2_nonadaptive_fp32 | gold | all | 501 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 |
| t5gemma2_nonadaptive_fp32 | gold | FAM_KIN | 207 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| t5gemma2_nonadaptive_fp32 | gold | MINOR_AGE | 166 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 |
| t5gemma2_nonadaptive_fp32 | gold | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 |
| t5gemma2_nonadaptive_fp32 | gold | GEN_NOUN | 96 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| t5gemma2_nonadaptive_fp32 | gold | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 |
| t5gemma2_nonadaptive_fp32 | loo | all | 501 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 |
| t5gemma2_nonadaptive_fp32 | loo | FAM_KIN | 207 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 |
| t5gemma2_nonadaptive_fp32 | loo | MINOR_AGE | 166 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 |
| t5gemma2_nonadaptive_fp32 | loo | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| t5gemma2_nonadaptive_fp32 | loo | GEN_NOUN | 96 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| t5gemma2_nonadaptive_fp32 | loo | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 |
| t5gemma2_nonadaptive_fp32 | rand | all | 501 | 0.026 | 0.088 | – | 0.000 | 0.031 |
| t5gemma2_nonadaptive_fp32 | rand | RANDOM | 501 | 0.026 | 0.088 | – | 0.000 | 0.031 |
| context-free prior (always top training spans) | any | all | 501 | 0.098 | 0.299 | 0.413 | – | – |
| context-free prior (always top training spans) | any | MINOR_AGE | 165 | 0.000 | 0.145 | 0.000 | – | – |
| context-free prior (always top training spans) | any | MINOR_EDU | 10 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior (always top training spans) | any | GEN_NOUN | 97 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior (always top training spans) | any | GEN_PHYS | 22 | 0.000 | 0.000 | 0.000 | – | – |
| context-free prior (always top training spans) | any | FAM_KIN | 207 | 0.237 | 0.609 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | all | 501 | 0.236 | 0.585 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | MINOR_AGE | 165 | 0.145 | 0.424 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | MINOR_EDU | 10 | 0.000 | 0.200 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | GEN_NOUN | 97 | 0.443 | 0.753 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | GEN_PHYS | 22 | 0.091 | 0.227 | 1.000 | – | – |
| type-oracle prior (top spans of the true type) | any | FAM_KIN | 207 | 0.237 | 0.691 | 1.000 | – | – |
