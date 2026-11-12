// Stub: Task 7 replaces this. Every number comes from data.json through v().
#let D = json("data.json")
#let v(k) = D.at(k)

The final test has #v("final.windows_n") windows; the richer model saves #v("final.diffs.rich_amt_baseline.mean|signed") on average.
