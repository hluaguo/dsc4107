#set page(paper: "us-letter", margin: 2.2cm)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em)[
  #set text(size: 15pt)
  #it
]

#align(center)[
  #text(size: 18pt, weight: "bold")[DSC 4107 Homework 1 Solutions]
]
#v(0.3em)
#align(
  center,
)[_Final answers are rounded to 4 significant figures. Intermediate values are kept unrounded._]

= Question 1

Let $F = 1000$, $y = 0.08$, $T = 3$.

*Q1-1. Annual coupon, $c = 10%$.* The coupon is $C = 0.10 times 1000 = 100$.

$
  P = sum_(t=1)^3 C/(1+y)^t + F/(1+y)^3 = 100 (1 - 1.08^(-3))/0.08 + 1000/1.08^3
$

$ P = 257.7096 + 793.8322 = 1051.54 approx bold(1052) $

#text(
  fill: rgb("#555"),
)[Calculator: N = 3, I/Y = 8, PMT = 100, FV = 1000 → CPT PV = −1051.54.]

*Q1-2. Zero coupon.* No coupons, so $C = 0$.

$ P = F/(1+y)^3 = 1000/1.08^3 = 793.83 approx bold(793.8) $

#text(
  fill: rgb("#555"),
)[Calculator: N = 3, I/Y = 8, PMT = 0, FV = 1000 → CPT PV = −793.83.]

*Q1-3. Quarterly coupon.* Per period: $C = 1000 times 0.10 \/ 4 = 25$, $y\/m = 0.02$, $n = 12$.

$
  P = 25 (1 - 1.02^(-12))/0.02 + 1000/1.02^12 = 264.3835 + 788.4932 = 1052.88 approx bold(1053)
$

This is higher than Q1-1 by about $1.33$ (1052.88 vs 1051.54). Quarterly coupons
arrive sooner than annual ones, and with the coupon rate (10%) above the yield (8%),
that earlier cash raises the price.

#text(
  fill: rgb("#555"),
)[Calculator: N = 12, I/Y = 2, PMT = 25, FV = 1000 → CPT PV = −1052.88.]

*Q1-4. Continuous compounding, $r = 8%$.* Discount each cash flow by $e^(-r t)$.

$
  P = 100 e^(-0.08) + 100 e^(-0.16) + 1100 e^(-0.24) = 92.3116 + 85.2144 + 865.2908 = 1042.82 approx bold(1043)
$

*Q1-5. Step-up coupon ($6%$, $8%$, $12%$).* Cash flows are $60, 80, 1120$.

$
  P = 60/1.08 + 80/1.08^2 + 1120/1.08^3 = 55.5556 + 68.5871 + 889.0921 = 1013.23 approx bold(1013)
$

*Q1-6. YTM of a $6%$ bond priced at 950.* Solve

$
  950 = 60/(1+y) + 60/(1+y)^2 + 1060/(1+y)^3, quad y = 7.938% approx bold(7.938%)
$

#text(
  fill: rgb("#555"),
)[Calculator: N = 3, PV = −950, PMT = 60, FV = 1000 → CPT I/Y = 7.938.]

= Question 2

Let $F = 100$, coupon rate $= 6%$ paid semi-annually, so $C = 3$. $y = 5%$, $m = 2$,
$y\/m = 0.025$, $n = 6$. Discount factor $"DF"_k = 1\/1.025^k$.

*Q2-1. Cash flow table.*

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, right, right, right),
  stroke: 0.4pt,
  table.header([*Period $k$*], [*Year*], [*CF*], [*DF*], [*PV of CF*]),
  [1], [0.5], [3], [0.975610], [2.926829],
  [2], [1.0], [3], [0.951814], [2.855443],
  [3], [1.5], [3], [0.928599], [2.785798],
  [4], [2.0], [3], [0.905951], [2.717852],
  [5], [2.5], [3], [0.883854], [2.651563],
  [6], [3.0], [103], [0.862297], [88.816577],
  [ ], [*Total*], [ ], [ ], [*102.754063*],
)

The current price is $P_0 = 102.7541$.

*Q2-2. Macaulay Duration* (weighted average time, in years):

$
  D_"Mac" = (sum_k (k\/m) times "PV"_k)/P_0 = 287.0119/102.7541 = bold(2.7932) " years"
$

*Q2-3. Modified Duration:*

$ D_"mod" = D_"Mac"/(1 + y\/m) = 2.7932/1.025 = bold(2.7251) $

*Q2-4. Meaning.* Modified duration measures price sensitivity to yield. $D_"mod" = 2.7251$
means a 1 percentage-point (100 bp) rise in yield lowers the price by about $2.73%$. As a
first-order approximation, it holds only for small yield changes.

*Q2-5. Approximate Modified Duration.* Re-price at $y = 5.1%$ and $y = 4.9%$:

$
  P_+ = 102.4745, quad P_- = 103.0345, quad P_0 = 102.7541, quad Delta y = 0.001
$

$
  "Approx. " D_"mod" = (P_- - P_+)/(2 P_0 Delta y) = 0.5600/0.205508 = bold(2.7251)
$

This matches the exact $D_"mod" = 2.7251$; a 10 bp move is small, and the linear
approximation holds well over that range.

= Question 3

Same bond: $F = 100$, $C = 3$, $m = 2$, $y = 5%$, $P_0 = 102.7541$, $D_"mod" = 2.7251$.

*Q3-1. Exact Convexity.*

$ C = 1/(P_0 m^2 (1+y\/m)^2) sum_(k=1)^6 "CF"_k/(1+y\/m)^k (k^2 + k) $

The weighted sum $sum "PV"_k (k^2+k) = 3920.6161$ and the denominator is
$P_0 times 2^2 times 1.025^2 = 431.8240$, so

$ C = 3920.6161/431.8240 = bold(9.0792) $

*Q3-2. Finite-difference convexity* with $Delta y = 0.01$ ($y = 6%$ and $4%$):

$ P_+ = 100.0000, quad P_- = 105.6014 $

$
  "Approx. Convexity" = (P_- + P_+ - 2P_0)/(P_0 (Delta y)^2) = 0.0933/0.0102754 = bold(9.0805)
$

This is close to the exact 9.0792. A 100 bp step is large enough that the two estimates
still differ a little.

*Q3-3. Yield rises 200 bp ($Delta y = +0.02$).*

Duration-only model:

$
  (Delta P)/P approx -D_"mod" Delta y = -(2.7251)(0.02) = -0.05450 = bold(-5.450%)
$

Duration + convexity model:

$
  (Delta P)/P approx -D_"mod" Delta y + 1/2 C (Delta y)^2 = -0.05450 + 1/2 (9.0792)(0.02)^2 = -0.05450 + 0.001816 = bold(-5.269%)
$

*Q3-4. Comparison.* The duration-only model predicts a loss of $5.450%$; adding convexity
gives $5.269%$. The gap is about $0.18$ percentage points. The true loss is
$(97.3357 - 102.7541)\/102.7541 = -5.273%$, so the convexity model is much closer.

Duration treats the price-yield relation as a straight line, while the actual curve is
convex and bows toward the origin. Over a large yield jump the line overstates the price
fall, and the positive convexity term corrects it. That asymmetry favors bondholders: for
equal yield moves, the price gain when yields fall is larger than the loss when yields rise.
