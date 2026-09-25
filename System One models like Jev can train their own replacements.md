---
title: System One models like Jev can train their own replacements
source: https://www.seangoedecke.com/system-one-models-can-train-their-own-replacements/
author: Sean Goedecke
published:
created: 2026-09-25
description: "“System One” models like Jev are fast general classifiers. Classifiers have existed since 1958, but they have to be trained for specific tasks: if you build a…"
tags:
  - clippings
---
“System One” models like [Jev](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/) are fast general classifiers. Classifiers have existed since [1958](https://en.wikipedia.org/wiki/Mark_I_Perceptron), but they have to be trained for specific tasks: if you build a classifier to identify images of dogs, it can’t be used to tell you if a streetlight is red, or if a letter is urgent. Like a LLM, Jev can be prompted for a wide variety of tasks, from [sorting email](https://www.youtube.com/watch?v=9oWxrsRo4d8) to [playing Doom](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/).

I think models like this are going to be important. There are many tasks that a LLM *could* do in theory but are too slow and expensive in practice (for instance, reading each new message in Slack [^1] and deciding whether to notify you or not). While you could train a specific classifier for these tasks, there are two main problems with that:

1. Despite being a well-understood ML problem, training a bespoke classifier is outside of the skillset of most ordinary engineering teams
2. Training a classifier requires assembling a large dataset

Jev obviously solves the first problem. Any engineering team can plug in a System One model with a prompt like “Based on {list of criteria}, should the user be notified about this message?” But I think it solves the second problem too.

For serious work, a specific hand-built classifier will always be cheaper and faster than Jev. Generic classifiers have to encode knowledge of all kinds of irrelevant things in their weights, so they can address lots of different tasks. That makes them larger, slower, and more expensive to run. Fortunately, **it is going to be surprisingly easy to replace a Jev instance with a hand-built classifier.**

Once you’re satisfied with how your Jev classifier is performing — presumably you’ve spent days tweaking the prompt — you can trivially collect its input and output data. In the Slack notifier case, that’d be the Slack message (plus any context) and the ultimate decision to notify or not. Once you’ve saved enough data, you’ll be able to train your own classifier on that data [^2].

It won’t be a general classifier like Jev, but it should do well on the specific task and be much faster. Of course it’ll require some ML expertise, but it should be easier to develop (or rent) that expertise once you’ve validated that the feature is worth building.

In other words, because Jev has to be prompted for specific tasks, it should be easy to [distil](https://en.wikipedia.org/wiki/Knowledge_distillation) any successful Jev usage into a specific classifier. If System One models take off — and I hope they do — I expect this to be a common pattern.

---

---

If you liked this post, consider [subscribing](https://buttondown.com/seangoedecke) to email updates about my new posts, or [sharing it on Hacker News](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fwww.seangoedecke.com%2Fsystem-one-models-can-train-their-own-replacements%2F&t=System%20One%20models%20like%20Jev%20can%20train%20their%20own%20replacements).

Here's a preview of a related post that shares tags with this one.

> Two techniques for working with System One models
> 
> I recently wrote about [Jev](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/), a new “System One” language model that only outputs *decisions*: the answers to a set of user-provided multiple-choice questions. This means it’s nowhere near as flexible as a traditional LLM like ChatGPT, but in return it’s consistently fast.
> 
> We don’t know exactly how Jev works. I’ve seen people say diffusion, or various tweaks to the Transformer architecture, or some entirely new type of model. But that doesn’t matter. Like I argued [here](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/#structured-output-can-already-be-fast), it isn’t hard to turn any LLM into a System One model. By batching prompts that generate a single token with structured output, you get a consistently fast general-purpose classifier. I vibed up a basic version to play with [here](https://github.com/sgoedecke/system-one/tree/main) in ~150 lines of Python (most of which is error handling).  
> [Continue reading...](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/)

---

[^1]: As I write this, I’m imagining ways you could poll and batch to do this with LLMs. Substitute “instantly notify” or some higher-volume event source if you’d prefer a different example.

[^2]: You could annotate a bunch of data with LLMs already, without using Jev, but this is a pretty expensive step to take when you aren’t sure the feature is going to work.