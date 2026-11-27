# Presentation script — Classic hedging is all you need

About 10 minutes, 13 slides. Times are cumulative.

---

## Slide 1 · Title (0:00–0:13)

Hi everyone. My project is called "Classic hedging is all you need." I compare
an old formula-based way of managing option risk with a newer neural-network way,
and put them head to head.

## Slide 2 · The hedging problem (0:13–1:12)

A bank sells a client a one-month call option and gets paid up front. If the
stock goes up a lot, the bank has to pay the client, so it's exposed to risk.

To protect itself, the bank holds some of the stock — this is called hedging. The
right number of shares changes as the price moves, so the bank keeps trading, and
every trade costs a small fee.

Trade often, and you're well protected but pay a lot of fees. Trade rarely, and
it's cheap but you're exposed.

This chart is from the first paper. Left to right is how many times you trade
over the month. Up and down is the Sharpe ratio. Sharpe is average profit divided
by how much the profit swings around — profit per unit of risk, higher is better.
Every curve peaks in the middle: there's a sweet spot.

## Slide 3 · Two papers, opposite ends (1:12–1:57)

Two papers attack this problem from opposite ends.

Sepp, in 2013, uses maths. He works out with formulas how often to trade, and
turns that into a simple rule: only trade when your position drifts too far from
where it should be. I'll call that "the band" — a rule with a tolerance band.

Buehler and co-authors, in 2019, use a neural network that learns by itself how
many shares to hold. It's more flexible — it can trade less, or only partly
protect itself.

Most people expect the network to win once trading gets expensive. But neither
paper tests both on the same data with the old rule properly tuned. I do.

## Slide 4 · Simulation framework (1:57–3:34)

Everything in this project runs in a simulation I built. Here's what one
simulated month looks like.

The stock starts at $100. The bank sells the option and receives $2.88. Then,
on each of the 21 trading days in the month, the strategy decides how many shares
to hold, and pays a fee of 0.2% on every trade. On the last day, the bank sells
whatever shares are left and pays the client whatever the option owes. The money
left over is the profit or loss for that month.

I test both methods under four market models: a smooth one, one with sudden jumps, one where the volatility moves around, and one with both.

Every strategy goes through the same accounting code. Training, tuning and the
final test each use separate data. And the classical band is re-tuned for every model and fee, so it never loses just because it was set up badly.

One detail in the box: the option is priced as if the stock moves 25% a year,
but it really moves about 20%. That gives the bank a small built-in margin, like
an insurer. So the real question is how much protection is worth paying for.

My main case is the moving-volatility model, called Heston, with a fee of 20
basis points. 20 basis points is 0.2% per trade — 20 cents on a $100 trade, which
is a realistic fee for trading a single stock.

## Slide 5 · Why Sharpe is not enough (3:34–4:16)

Before the results, one thing to note: Sharpe can be fooled. A strategy that simply stops protecting itself and keeps the $2.88 can look good on Sharpe while taking huge risk of option lost.

A bank cares about two things. First, mean cost: how much it pays in trading fees
on average. Second, tail loss: in the worst 5% of possible futures, how much it
loses on average. Think of it as "how bad are the bad days?"

So my rule is strict. The network only counts as better if it is cheaper and
safer than the classical band — both at once.

## Slide 6 · Replication 1: Sepp (4:16–4:47)

First I checked that my code reproduces each paper.

This table puts my numbers next to Sepp's: the best number of trades per year,
and the Sharpe you get there. They match closely.

I also checked the formulas against 60,000 simulated price paths, and they
agree. And the pattern is clear: there's always a sweet spot, and the more
complicated the market — jumps, moving volatility — the less often you should
trade. So the classical side is trustworthy.

## Slide 7 · Replication 2: deep hedging (4:47–5:24)

For the neural network, there's a simple check. With zero fees, the correct answer
is already known — it's the textbook Black-Scholes hedge. My network learned
almost exactly that: a 0.99 correlation.

Then I add fees: 20, 50, and 100 basis points. At 20 basis points, the network
scores a Sharpe of 0.37, against 0.25 for the simple strategy of trading at every
step. As fees rise, the network stays ahead — that's what the paper claims.

So both copies work. But does the network still win against a properly tuned
classical rule?

## Slide 8 · 20 bp: the headline (5:24–6:10)

This is the main result. There's one panel per market model. Blue is the
network's Sharpe, orange is the best classical rule's — re-tuned for every panel and fee, and the fee goes up from left
to right.

Look at the green strip — 20 basis points, my main test. In every panel, the two
lines sit right on top of each other. In the two models with jumps, orange is
even slightly ahead. So at realistic fees, it's basically a tie.

I then ran one final test on five fresh batches of data I had never touched. On
the Heston model the network won by 0.034 in Sharpe. That's real, but tiny. So
does it pass my rule — cheaper and safer?

## Slide 9 · Same spread, worse tail (6:10–6:46)

No. This chart is in dollars per option sold.

On the left is the spread — how widely the outcomes vary. It's almost the same
for both, so on a normal day they look alike.

On the right is the tail loss. In the worst 5% of futures, the network loses
$1.32 per option on average. The band loses $1.07. That's 24% worse, and it's
worse on every one of the five final test batches.

So the network's small Sharpe win is paid for with worse bad days.

## Slide 10 · Can a safer-trained network beat the band? (6:46–7:52)

A fair objection: the network was trained mostly on average outcomes. So I retrained it to care about bad days, in two ways. One tells it to focus only on its worst outcomes. The other adds a penalty during training whenever its bad days are worse than the band's. Each blue dot is
one retrained network.

Here's how to read the chart. Left is cheaper. Down is safer. The red star is the
Sepp band, my benchmark. The orange dots are the other classical rules. To beat it, a network has to land in the green corner — cheaper
and safer.

That corner is empty. Every network is either cheaper but riskier, or safer but
more expensive.

There is one niche. The dashed line is the safest any classical rule gets. Some
networks go below it — safer than any classical rule — but they pay about 8% more in fees.

## Slide 11 · The high-cost twist (7:52–8:38)

Now back to the right side of the earlier chart. When fees get very high — 100 to
200 basis points, which is 1 to 2% per trade — the classical rule's Sharpe goes
negative, but the network stays positive. At 100 basis points it's 0.18 against
minus 0.25. That looks like a big win.

But look at the spread columns. At 100 basis points, the network's outcomes spread
3.11 dollars, against 1.73 for the classical rule — nearly double. What's happened
is the network has learned to stop protecting itself and just keep the $2.88.
That looks good on Sharpe, but it isn't hedging any more. It's a different, much
riskier bet.

## Slide 12 · Caveats (8:38–9:11)

A few limitations. My computing resources were limited — small networks and
limited training — so the exact numbers are less precise than a full research
paper's. But the direction held when I let an automatic search tune the network.

Also, in the final test the network was an average of three trained networks
against a single classical rule. That actually favours the network.

On the right: I repeated the test at 10 basis points, a fee closer to very liquid
index products. One training recipe does narrowly win there — about 0.4% cheaper —
but only at one setting and on fewer than half of the random seeds. It's too
fragile to change the headline.

## Slide 13 · Conclusion (9:11–9:41)

To sum up.

One: at realistic fees, the network's advantage is tiny and comes with worse bad
days. Nothing beats the classical band on both cost and safety.

Two: the network is not useless. Trained to focus on bad days, it reaches a
safer tail than any classical rule — 0.94 against 0.99 — if a desk is willing to
pay for it.

Three: big Sharpe wins at high fees come from not hedging at all — so always
report the spread and the tail loss alongside Sharpe.

And both papers' results reproduce.
So my answer: classic hedging is all you need. Thank you.

---

## Quick reference for questions

| Term                     | Plain meaning                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Call option              | Client's right to buy the stock at $100 in a month; the bank pays if the price rises |
| Premium                  | The $2.88 the bank is paid up front for selling the option                           |
| Hedging                  | Holding shares to cancel out the option's risk                                       |
| Basis point (bp)         | 0.01%. 20 bp = 0.2%, 100 bp = 1%, 200 bp = 2%                                        |
| Sharpe ratio             | Average profit ÷ how much profit swings. Higher is better                            |
| Standard deviation (std) | How widely outcomes spread                                                           |
| Tail loss (CVaR95)       | Average loss in the worst 5% of outcomes. Lower is safer                             |
| Skew                     | Negative = bad surprises bigger than good ones                                      |
| The band                 | Sepp-style rule: trade only when the position drifts too far                         |
| Alpha (α)                | Share of worst outcomes the network trains on. 0.65 ≈ average; 0.05 = worst 5% only  |

---

## Likely questions and how to answer

Keep answers short: one sentence of answer, one sentence of evidence, stop.

### About the setup

**Q: Why use simulated data instead of real stock prices?**
A: Real history gives you only one path, so you can't measure a worst-5% loss
reliably from it. Simulation gives me hundreds of thousands of possible futures,
and I know the true market model, so any difference comes from the strategy, not
from luck in the data.

**Q: Is 20 basis points realistic?**
A: It's in the realistic range for trading an individual stock once you include
the bid-ask spread. Very liquid index products are cheaper, which is why I also
tested 10 basis points, and the conclusion there was the same: no reliable win
for the network.

**Q: Why price the option at 25% volatility when the stock moves 20%?**
A: Banks really do charge a bit more than expected volatility, the same way an
insurer charges more than expected claims. Without that margin, never hedging
would score best on Sharpe, and the comparison would just be "who trades least".

**Q: Why only 21 trading days?**
A: It's one trading month, so each strategy can rebalance at most once a day,
which is the realistic setting for an option desk.

### About the neural network

**Q: What does the neural network actually look like?**
A: A small network with three hidden layers of 64 units (128 for the final test). Each day it sees three
things: where the stock is relative to the strike, how much time is left, and how
many shares it currently holds. It outputs how many shares to hold next.

**Q: How was it trained?**
A: On 2,048 simulated months per training run, for 80 to 100 epochs (200 for the final test). It is
trained to minimise the average of its worst outcomes, which is the objective
from the Buehler paper. Every result averages three separately trained networks.

**Q: Maybe a bigger network or more training would win?**
A: Possibly, and I list that as a limitation. But the direction didn't change
when I ran an automatic hyperparameter search: at 20 basis points the result was
still a tie. And at zero fees the network already finds the right answer, so it
isn't failing to learn.

**Q: What does "0.99 correlation with the textbook hedge" mean?**
A: When there are no fees, the theoretically correct number of shares is known.
The network's choices line up with it almost perfectly, which shows the network
and my simulation are working correctly.

### About the results

**Q: The network did win on Sharpe in the final test. Why isn't that a win?**
A: Because the win comes with a 24% worse loss on the bad days. A bank would not
accept a slightly better average in exchange for bigger disasters, so I require
it to be both cheaper and safer.

**Q: How do you know the +0.034 isn't just luck?**
A: It was positive on all five test batches — even the smallest was 0.029.
So it's a real edge on unseen data, just a very small one. One caveat: all five
batches were scored by the same trained networks, so retraining on fresh data could
move it by more than that.

**Q: Isn't the network's worse tail just because you trained it on averages?**
A: That's exactly the objection slide 10 answers. When I retrained it to focus on
bad days, it could match the band's tail, but only by paying more in fees. It
never got both.

**Q: So is the neural network useless?**
A: No. It's a different option, not a better one. If a bank wants a safer result
than any classical rule can give, the network can deliver that, at about 8% more
in fees.

**Q: Why does the network stop hedging when fees are high?**
A: Because at 1 to 2% per trade, the fees cost more than the protection is worth,
and the bank still keeps the option fee. The classical rules can't make that
choice — they always buy the full hedge on day one — so part of the network's
high-fee edge is a choice the classical side doesn't have. When I added a
classical rule that can hedge only part of delta, its best setting at those fees
was zero: don't hedge. So it's the rational move, but it's no longer hedging.

**Q: How is the classical rule tuned? Isn't that unfair to the network?**
A: For each market model and fee, I try a set of classical rules on separate
tuning data and keep the best one. The network gets its own tuning too, and the
final test uses data neither of them has seen.

**Q: You compared three averaged networks against one rule. Is that fair?**
A: It actually favours the network, because averaging smooths out noise. Even
with that advantage, it didn't pass the cheaper-and-safer test.

### About the replications

**Q: Your Sepp numbers don't match exactly. Why?**
A: They're within a few percent. Most of the Sharpe gap is one convention: Sepp's
text says to subtract the cost of the opening trade, and I do, but his printed
Sharpe column leaves it out. Without that adjustment I get his numbers (0.90 vs
0.89, 1.30 vs 1.30). I also checked my formulas against 60,000 simulated paths,
which agree with them.

**Q: Did you reproduce the deep hedging paper's exact numbers?**
A: I reproduced its main claim: the network beats trading every day once fees are
added. My setup is smaller than theirs because of limited computing resources, so
I focus on the pattern, not the exact figures.

### Big picture

**Q: What should a bank take away from this?**
A: At normal fees, a well-tuned classical rule is as good as a neural network,
and it's simpler and easier to explain. And whenever someone shows a Sharpe win,
check the spread and the worst-case losses too.

**Q: What would you do next with more time?**
A: Train bigger networks on more data, and test on real market data to see
whether the same pattern holds.
