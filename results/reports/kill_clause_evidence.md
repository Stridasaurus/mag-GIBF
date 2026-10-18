# Kill-clause evidence (facts only; the reading is Strider's)

ROADMAP §5-D2: method-wide abandonment needs all four of (1) D2 = drop, (2) no advantage at nonzero φ in the coherent regime, (3) no B6 advantage under misspecification, (4) B4 attributing the acoustic-GIBF gap to absent kernel phase diversity. What the records say about each part:

## Part 1: D2 outcome (sealed adjudication)

- WIN rule met: False
- LOSS rule met: True (at φ = [180.0])
- B6 rule met: False (winner None, fragile True)

## Part 2: signed gap (GIBF minus MMV, mean Δr̄) at d = 2, nonzero φ, by SNR
Negative favours GIBF. 'sealed' cells are confirmatory; 'completion' cells are descriptive.

| φ | SNR | gap | 95% CI | source |
|---|---|---|---|---|
| 30 | -5 | 1.481 | [1.352, 1.620] | completion |
| 30 | 0 | 0.058 | [0.001, 0.116] | completion |
| 30 | 5 | 0.031 | [0.015, 0.049] | completion |
| 30 | 10 | 0.129 | [0.123, 0.134] | completion |
| 30 | 20 | 0.077 | [0.074, 0.081] | completion |
| 60 | -5 | 0.779 | [0.615, 0.936] | completion |
| 60 | 0 | -0.277 | [-0.377, -0.174] | completion |
| 60 | 5 | 0.644 | [0.568, 0.722] | completion |
| 60 | 10 | 0.586 | [0.526, 0.650] | completion |
| 60 | 20 | 0.051 | [0.045, 0.057] | completion |
| 90 | -5 | -0.763 | [-0.941, -0.582] | completion |
| 90 | 0 | -0.345 | [-0.496, -0.192] | completion |
| 90 | 5 | 3.371 | [3.207, 3.520] | sealed |
| 90 | 10 | 5.239 | [5.068, 5.416] | sealed |
| 90 | 20 | 7.295 | [7.117, 7.474] | completion |
| 120 | -5 | -0.260 | [-0.365, -0.150] | completion |
| 120 | 0 | 0.141 | [0.108, 0.174] | completion |
| 120 | 5 | 0.069 | [0.064, 0.074] | completion |
| 120 | 10 | 0.071 | [0.067, 0.074] | completion |
| 120 | 20 | -0.023 | [-0.028, -0.019] | completion |
| 150 | -5 | 0.055 | [0.044, 0.067] | completion |
| 150 | 0 | 0.081 | [0.077, 0.085] | completion |
| 150 | 5 | 0.063 | [0.060, 0.067] | completion |
| 150 | 10 | 0.041 | [0.039, 0.044] | completion |
| 150 | 20 | -0.000 | [-0.002, 0.001] | completion |
| 180 | -5 | 0.201 | [0.195, 0.207] | completion |
| 180 | 0 | 0.227 | [0.222, 0.233] | completion |
| 180 | 5 | 0.267 | [0.255, 0.281] | sealed |
| 180 | 10 | 0.584 | [0.532, 0.637] | sealed |
| 180 | 20 | 1.147 | [1.062, 1.238] | completion |

## Part 3: B6 (sealed)

- {"met": false, "winner": null, "by_favored": {"gibf": {"met": false, "effect_met": false, "ci_non_overlap": false, "favored_mean": 6.531848066435395, "other_mean": 5.456546272291495, "favored_ci": [6.36703085890941, 6.698215273961379], "other_ci": [5.29977639395641, 5.6202303181545785]}, "mmv": {"met": false, "effect_met": false, "ci_non_overlap": true, "favored_mean": 5.456546272291495, "other_mean": 6.531848066435395, "favored_ci": [5.29977639395641, 5.6202303181545785], "other_ci": [6.36703085890941, 6.698215273961379]}}, "fragile": true, "confirmatory": true, "delta_r_arm": {"met": false, "effect_met": false, "ci_non_overlap": false, "favored_mean": 6.531848066435395, "other_mean": 5.456546272291495, "favored_ci": [6.36703085890941, 6.698215273961379], "other_ci": [5.29977639395641, 5.6202303181545785]}}

## Part 4: B4 d_min (smallest d with P_sep >= 0.8), GIBF, by SNR / kernel / reduction

```
{
 "-5.0|real|on": 5.0,
 "-5.0|real|off": 5.0,
 "-5.0|lit|on": 5.0,
 "-5.0|lit|off": 5.0,
 "-5.0|w1|on": 3.0,
 "-5.0|w1|off": 3.0,
 "-5.0|w2|on": 2.0,
 "-5.0|w2|off": 2.0,
 "-5.0|w5|on": 2.0,
 "-5.0|w5|off": 2.0,
 "0.0|real|on": 3.0,
 "0.0|real|off": 3.0,
 "0.0|lit|on": 3.0,
 "0.0|lit|off": 3.0,
 "0.0|w1|on": 3.0,
 "0.0|w1|off": 3.0,
 "0.0|w2|on": 2.0,
 "0.0|w2|off": 2.0,
 "0.0|w5|on": 2.0,
 "0.0|w5|off": 2.0,
 "5.0|real|on": 2.0,
 "5.0|real|off": 2.0,
 "5.0|lit|on": 2.0,
 "5.0|lit|off": 2.0,
 "5.0|w1|on": 3.0,
 "5.0|w1|off": 3.0,
 "5.0|w2|on": 2.0,
 "5.0|w2|off": 2.0,
 "5.0|w5|on": 2.0,
 "5.0|w5|off": 2.0,
 "10.0|real|on": 2.0,
 "10.0|real|off": 2.0,
 "10.0|lit|on": 2.0,
 "10.0|lit|off": 2.0,
 "10.0|w1|on": 3.0,
 "10.0|w1|off": 2.0,
 "10.0|w2|on": 2.0,
 "10.0|w2|off": 2.0,
 "10.0|w5|on": 2.0,
 "10.0|w5|off": 2.0,
 "20.0|real|on": 3.0,
 "20.0|real|off": 3.0,
 "20.0|lit|on": 3.0,
 "20.0|lit|off": 3.0,
 "20.0|w1|on": 3.0,
 "20.0|w1|off": 2.0,
 "20.0|w2|on": 2.0,
 "20.0|w2|off": 2.0,
 "20.0|w5|on": 2.0,
 "20.0|w5|off": 2.0
}
```
