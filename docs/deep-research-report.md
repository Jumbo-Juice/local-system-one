# Joint Embedding Vectors (JEV) – Executive Summary

**Joint Embedding Vectors (JEV)** are the core of TypeSafe AI’s new *System One* models – a class of ultra-fast, structured-decision models that output probabilities instead of text.  Unlike traditional LLMs, Jev consumes unstructured state (e.g. text or JSON) plus a set of predefined *questions* (Choice, Score, Noul) and returns **typed answers with calibrated probability distributions**.  This design eliminates hallucination (answers cannot stray outside the allowed options) and makes outputs immediately usable in code.  TypeSafe claims Jev matches frontier-LLM accuracy on such tasks while being tens to hundreds of times faster (≈70–500 ms per call) and much cheaper (input token pricing ~$0.042/M vs $0.20–10/M for LLMs).  In short, JEV models trade natural-language generation for **deterministic, low-latency classification**.  This report provides a complete technical dossier on JEV: definitions, goals, architecture, data and training, implementation guidance, evaluations, deployment, and legal considerations, all with citations to primary sources.  

## Key Concepts and Goals

- **Structured Decisions, Not Text**: TypeSafe’s manifest explains that large-scale automation needs *“type-safe structured values”* rather than free-form text.  In practice, Jev questions are like software assertions: e.g. “Which department should handle this ticket?” with explicit answer choices.  The model **never generates strings**; it outputs exactly the requested answer type with probabilities.  

- **Calibrated Probabilities (RLCD)**: A central goal is *calibration* – the model’s confidence equals its true accuracy.  TypeSafe invented a new training regime called **Reinforcement Learning for Calibrated Decisions (RLCD)**.  Instead of optimizing human preferences (RLHF) or hard-coded rewards (RLVR), RLCD optimizes for correct decisions *and* correct confidence levels.  In theory, events assigned probability *p* occur about *p* of the time.  Good calibration is crucial for automation: code can branch on confidence (e.g. “if Jev’s confidence <90%, escalate to a human”).  

- **System One vs System Two**: TypeSafe frames Jev as **System One** intelligence (fast, intuitive decisions) in contrast to slow reasoning LLMs.  Jev excels at quick "gut-check" judgements on a rich state (like classification, scoring, sorting) and is meant to plug into software workflows, whereas System Two LLMs generate prose or perform complex chain-of-thought tasks.  A key design is **no human-in-the-loop**: Jev answers slot directly into code as fuzzy `if/else` rules.  

- **Output Primitives**: TypeSafe defines three question types: 
  - **Choice**: select one among named options (returns `choice`, `probabilities`, `confidence`).  
  - **Score**: rate on an ordered rubric (returns `score`, `probabilities`, `confidence`, plus the level labels).  
  - **Noul**: a yes/no judgment (returns a single probability `noul` between 0–1).  
  All answers are constrained to the supplied categories; the model *cannot* return an unknown option.  

These features make Jev a **drop-in AI decision service** for software: you define a JSON *state*, list your questions and their finite answer sets, and Jev returns a JSON answer object with chosen values and calibrated confidences. 

## Architecture and Algorithm

### Transformer Backbone with Parallel Heads

Jev’s exact model architecture is proprietary, but it is reported to be **transformer-based** and loosely derived from open LLMs.  The key changes from a chat LLM are at the output stage:

- **Parallel Inference**: Jev uses a *parallel sampler* – it generates all question outputs in one forward pass, instead of autoregressive token-by-token sampling.  In effect, the model encodes the state and question instructions, then simultaneously produces probability distributions for each answer.  This makes inference extremely hardware-efficient (maximising GPU parallelism).  

- **Multiple Output Heads**: Conceptually, Jev can be thought of as having separate “heads” or classification layers for each question.  For a Choice question, one head outputs a softmax over the  *n* provided options.  For a Score question, a head outputs a distribution over the ordered rubric levels.  For a Noul question, it outputs a single value (Yes probability).  All heads operate on the same encoded state representation.  TypeSafe guarantees that the model *“never makes type errors”* – it can only output the valid schema fields.  

- **Joint Embedding (Hypothetical)**: Although not publicly detailed, “joint embedding vectors” suggests Jev may embed **state and each answer option into a shared space**.  One plausible design is: encode the state+question into a vector, and also embed each candidate answer (or rubric label).  Then compute dot-products or similarity scores to get logits for each answer (a bit like CLIP-style models embed text/image pairs).  This would allow scoring all options in parallel.  (Indeed, an open-source analog *OpenJev* uses a cross-encoder approach: it encodes the state and each option in the context of one forward pass per option.)  The term “joint embedding vectors” likely refers to these combined representations of (state,answer).  

- **High Cardinality Handling**: Jev supports up to **255 choices per question**.  If more options are needed, it uses a two-stage approach: first it independently *scores* batches of options, then does a final choice among top candidates.  This keeps most queries fast but incurs a slowdown for very large option sets.

### Input Representation

- **Tokenisation**: The state is arbitrary unstructured data (free text or JSON) but is tokenised in the model’s usual way (likely byte-pair or sentencepiece as with the base transformer).  Structured JSON fields can be included directly; for example, cloudflare’s docs show passing a Python dict as `state` (with nested keys like `ticket.message`).  The model apparently understands the backtick markup to refer to state fields in the questions.  

- **Context Window**: TypeSafe’s published model accepts very long inputs.  One source says a context window of **32,000 tokens** (some sources claimed 64k, but Cloudflare docs cite 32k).  This allows large documents or logs to be evaluated.  

- **Question Formatting**: Each question has an `id`, a `type` and `instructions` (the human-readable prompt), plus `criteria` listing the options/levels.  These criteria become the only possible outputs.

### Calibration and Output

Because Jev is trained to calibrate, it returns not only probabilities but a separate **confidence** score for Choice/Score questions.  The `confidence` is a scalar summary of the peakiness of the distribution.  For Noul questions, the output itself (0–1) *is* the probability of “yes” (i.e. a calibrated confidence).  Empirically, higher returned probability **correlates with accuracy**.

## Training Data and Objectives

TypeSafe has not publicly detailed Jev’s training dataset.  The blog’s FAQ asks “Where does our training data come from?” but gives no answer. We can infer likely sources and methods:

- **Possible Data Sources**:  Jev must be fine-tuned or trained on **structured evaluation tasks**.  This could include existing classification datasets (sentiment, topic, spam detection, etc.), NLI (entailment datasets), and custom Q&A tasks.  The OpenJev example shows that training on *NLI-style* entailment (premise/hypothesis) gives a generic decision model.  It’s plausible TypeSafe aggregated many such tasks, perhaps including code-review tasks, extracted QA pairs, or even logs of LLM calls with known outcomes.  

- **Preprocessing**: Each training example would include a state (text or structured record) and one or more questions with correct answers.  The state and the question (instructions + choices) are concatenated in a consistent format for the model.  The criteria/options are tokenized and possibly embedded as special tokens or text spans.  The model never sees any new output text, only the fixed answer labels (so training labels are exactly the provided options). 

- **Training Objective**: The backbone model is likely first pretrained (e.g. on language modeling) and then **fine-tuned** under RLCD.  A plausible approach: for each (state,question), maximize log-probability of the correct answer (cross-entropy loss) *and* apply a secondary calibration reward.  For example, treat the model’s predicted probability of the correct answer as a reward signal in an RL loop, encouraging accurate confidence.  Alternatively, use techniques like isotonic regression or temperature scaling on the probabilities.  Without detailed disclosure, we note only that RLCD implies reinforcement learning post-processing. 

- **Loss Functions**: We expect categorical cross-entropy for choice/score tasks.  Calibration could be enforced by an auxiliary loss (e.g. Brier score or KL divergence to a “one-hot” target distribution).  RLHF-like policy gradient might also be used if a reward is defined for calibration (e.g. reward = 1.0 if answer correct, weighted by predicted prob). 

- **Tokenisation and Embeddings**: The model uses standard token embeddings for words or JSON tokens.  Embedding sizes and architecture depth are undisclosed.  TypeSafe suggests Jev’s base is an *open-weight LLM* (“outside observers suspect”), possibly on the order of a few billion parameters.  (For example, the OpenJev project fine-tuned a 4B-parameter model for Jev tasks.) 

- **Multi-Modal Handling**: Currently Jev is text/JSON-only.  One blog notes both Jev and Qwen demos were “text-only” and adding image input would require a different model.  In principle, one could fine-tune a multi-modal foundation (like a vision+language model) to output structured decisions, but TypeSafe has not announced an image-capable JEV as of 2026.

## Implementation Guide (for Claude-Compatible Integration)

Below is a step-by-step plan to implement and integrate a JEV-style model into an AI system or agent (e.g. an Anthropic Claude agent).

1. **Choose an Approach** – Option A: **Use TypeSafe’s Jev API** (Easiest). Option B: **Build/Self-Host** an open alternative (like *OpenJev*) if API access is not desirable.  
   - *TypeSafe API*: Sign up for Jev early access or enterprise. Get API key. Use the official client (TypeSafe SDK or cloud APIs). (See Python example below.)  
   - *OpenJev (Self-Hosted)*: Use an open model like [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) on HuggingFace. This project provides a Qwen-based Jev which you can download (checkpoints ~4B parameters) under an MIT license. Ensure you have sufficient GPU (the code example needed an H100 for speed tests).  

2. **Define State and Questions** – Decide what “state” your agent will send. This could be user input, a document, or any context (as free text or structured JSON). Define one or more *questions* (Choice/Score/Noul) relevant to your task, each with a unique ID, instructions, and full list of options or rubric levels.  Example (English): 
   ```json
   {
     "state": "The user wrote: 'The server is down and customers are complaining loudly.'",
     "questions": {
       "issue": {
         "type": "choice",
         "instructions": "What is the category of this support ticket?",
         "criteria": {
           "outage": "Service outage",
           "billing": "Billing or account",
           "other": "Other issue"
         }
       },
       "priority": {
         "type": "score",
         "instructions": "How urgent is this ticket?",
         "criteria": ["Low", "Medium", "High"]
       }
     }
   }
   ``` 

3. **Query the Model** – Send the state and questions to the Jev model.  If using TypeSafe’s service, the API endpoint is:
   ```
   POST https://api.typesafe.ai/v1/systemone
     Authorization: Bearer YOUR_API_KEY
     Content-Type: application/json
   ```
   with a body like the JSON above.  The response will be a JSON with each answer’s type, chosen value, probabilities, and confidence.  

   *Python example (TypeSafe SDK):*
   ```python
   from typesafe_sdk import Choice, Score, Noul, TypeSafeClient

   client = TypeSafeClient()  # reads API_KEY from environment
   questions = {
       "issue": Choice(
           instructions="What is the category of this support ticket?",
           criteria={"outage": "Service outage", "billing": "Billing issue", "other": "Other issue"}
       ),
       "priority": Score(
           instructions="How urgent is this ticket?",
           criteria=["Low", "Medium", "High"]
       )
   }
   state_text = "The server is down and customers are complaining loudly."
   response = client.system_one(state=state_text, questions=questions)
   # response.answers["issue"].choice, response.answers["issue"].confidence, etc.
   ```
   This example is adapted from TypeSafe’s [Quick Start guide].  

   If *self-hosting* with OpenJev (or your own model), you must implement a function that feeds (state,question,option) into the model and extracts probabilities.  The OpenJev approach is: for each *option*, run one forward pass (premise=state+question, hypothesis=option) and read out the probability of “entailment” as the score for that option.  Finally take a softmax across options.  (This is slower than TypeSafe’s one-shot inference but can be batched or parallelised.)  

4. **Use the Results** – The returned answers include: 
   - For **Choice**: `choice` (the top label), `probabilities` (map from label→prob), and `confidence` (0–1).  
   - For **Score**: `score` (numeric index or value), `legend` (mapping of indices to your labels), `probabilities`, `confidence`.  
   - For **Noul**: a single number `noul` (0–1, where e.g. 0.95 means “Yes” with 95% confidence).  

   In your code (e.g. Claude agent or microservice), branch on these.  For example:
   ```python
   ans = response.answers["priority"]
   if ans.score >= 2:   # “High” in legend index 2
       escalate_ticket()
   if ans.confidence < 0.7:
       defer_to_human("Low confidence on ticket analysis")
   ```
   Always verify that the chosen answer fits your logic; since Jev outputs are type-safe, your code need not do heavy parsing.  

5. **Composing multiple questions** – Jev supports batching questions in one call: each question is answered in parallel.  This has near-constant latency regardless of number of questions (up to the model’s capacity).  To combine answers, simply use your own logic after receiving all answers.  For instance, you might multiply a `Noul` probability by a `Score` weighting, or trigger a follow-up only if two confidences are high.  TypeSafe recommends splitting complex judgments into atomic questions (each a “snap judgment”) and then composing the results in code.  

6. **Claude-Agent Integration** – TypeSafe provides a [Claude Skill](https://github.com/typesafe-ai/skills) for Claude Code and LangChain.  With it, you can add Jev as an AI tool.  For example, in a Claude prompt: 
   ```
   *Tool:* Ask TypeSafe Jev system questions by using the `system_one` function. Define your questions and criteria as shown. The tool returns JSON with fields 'choice','score','noul','confidence'. Use these to branch in the code.
   ```
   (If your framework supports plug-ins, follow the instructions in the Quickstart to install the TypeSafe skill.)  Essentially, treat Jev as an AI Oracle: your Claude-based agent formulates the question JSON, calls the Jev API (via the SDK), and then uses the numeric answers to guide its next actions.

```mermaid
graph LR
  A[State (text or JSON)] -->|“evaluate state”| B[Jev Model]
  B -->|question1| C[Choice Head ➔ answer + prob + confidence]
  B -->|question2| D[Score Head ➔ score + prob + confidence]
  B -->|question3| E[Noul Head ➔ yes-probability]
  C --> F[Use in code (branches, logic)]
  D --> F
  E --> F
```

## Example Code Snippet

Below is a Python sketch for a custom Jev-like loop (using an open model). This illustrates one way to build parallel decision queries if **not** using the TypeSafe API.  It assumes you have a transformer-based model `jev_model` (e.g. from Hugging Face) and a tokenizer `tokenizer`:

```python
import torch

def run_jev(jev_model, tokenizer, state, questions):
    answers = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            # Encode state and question instructions once
            context = state + " " + q["instructions"]
            inputs = tokenizer(context, return_tensors="pt")
            # Score each option
            logits = []
            for label in q["criteria"].keys():
                inpt = tokenizer(label, return_tensors="pt")
                # Simplest: cross-encode [context + label], or use a joint embedding scheme
                combined = {k: torch.cat([inputs[k], inpt[k]], dim=1)
                            for k in inputs.keys()}
                out = jev_model(**combined)
                logits.append(out.logits)  # assume single logit for entailment
            probs = torch.softmax(torch.tensor(logits).view(-1), dim=0)
            best = torch.argmax(probs).item()
            label = list(q["criteria"])[best]
            answers[qid] = {"choice": label,
                            "probabilities": {opt: float(p) for opt,p in zip(q["criteria"], probs)},
                            "confidence": float(probs[best])}
        elif q["type"] == "noul":
            # Example: single forward pass for yes/no (binary)
            prompt = state + " " + q["instructions"]
            out = jev_model(**tokenizer(prompt, return_tensors="pt"))
            prob_yes = float(torch.sigmoid(out.logits)[0])
            answers[qid] = {"noul": prob_yes}
        # (Add Score handling similarly)
    return answers

# Usage:
state = "My refund was delayed and I am upset."
questions = {
    "urgent": {"type": "noul", "instructions": "Does this message express urgency?"},
    "topic": {"type": "choice", 
              "instructions": "What is this about?",
              "criteria": {"billing": "Billing issue", "refund": "Refund request", "other": "Other"}}
}
response = run_jev(jev_model, tokenizer, state, questions)
print(response)
```

This pseudocode demonstrates the **joint evaluation** approach: encoding the state once and scoring each option. (In practice, one would optimize batching and use the model’s actual classification API.) 

## Evaluation and Benchmarks

**Benchmarks:** TypeSafe and the community have begun evaluating JEV models on classification tasks.  TypeSafe published some *in-house* results: Jev processed a representative workflow in 0.114 s vs 8.566 s for LLMs (≈193× faster), at roughly $0.00008 vs $0.0139 per workflow (≈444× cheaper).  Accuracy was ~67.8% (Jev) vs 67.9% (OpenAI GPT-5.6 Terra) and 67.8% (Claude) on that suite.  They also report Jev’s end-to-end response time is ~70–500ms vs 3–329s for LLMs.  No independent academic benchmarks exist yet, but community projects are emerging: for example, the **JevBench** benchmark evaluates structured QA tasks, and an open-source Jev model (based on Qwen3.5) achieved scores close to the proprietary Jev on public tasks.  

**Metrics:** Key metrics for JEV include:
- **Accuracy** on the target questions (same as any classifier).
- **Calibration** error (e.g. expected vs. actual frequencies of predicted probabilities).
- **Throughput/Latency:** type-safe tasks per second.  TypeSafe claims 6–14 queries/sec per GPU (e.g. Doom demo 10 q/s on an H100).
- **Cost/Efficiency:** input tokens (state length) and any output processing cost (output is free for Jev, per TypeSafe).
- **Robustness/Ablations:** There are no published ablations of Jev architecture, but one can vary question count, choice count, etc.  For very large choice sets (>255), accuracy may degrade, so TypeSafe suggests hierarchical scoring.

**Trade-offs:** The main trade-off is **flexibility vs reliability**.  By fixing outputs, Jev is limited to the questions you define; it cannot invent new labels or perform open-ended reasoning.  In exchange, it gains speed and predictability.  Table:  
| Model            | Output Type         | Latency          | Cost per 1k input tokens | Hallucinations   | Calibrated?    | License        |
|------------------|---------------------|------------------|--------------------------|------------------|----------------|----------------|
| **TypeSafe Jev** | Structured JSON (Choice/Score/Noul) | 0.07–0.5 s | ~$0.042 (input) | None (schema-locked) | Yes | Proprietary (API) |
| **OpenJev (Qwen3.5)** | Same struct. (via entailment) | ~0.5–1 s (per option, 4B model) | Free (open source) | None (finite options) | Possibly (tuned on entailment) | MIT (open source) |
| **LLM (e.g. GPT-4)** | Free text (then parsed) | 3–300 s | $\$2$–$50$ per 1k input| Often (can hallucinate) | No (overconfident) | Closed (OpenAI, etc) |

Table: **Comparison of Jev (TypeSafe), OpenJev, and a typical LLM**.  All figures approximate; see sources.  

## Deployment and Inference

- **Inference (GPU/CPU)**: Jev can be hosted via TypeSafe’s cloud or on-prem with adequate GPUs.  The Akka blog notes a local OpenJev model (4B params) can run 10–15 q/s on an H100 or top-end GPU.  CPU-only would be much slower; a GPU is recommended for real-time use.  

- **Latency**: Expect ~70–500ms per API call (for TypeSafe Jev).  This low latency enables sub-second decisions in apps.  Batch multiple questions to amortize this latency, since multiple questions cost virtually no extra time.  

- **Quantisation**: To reduce resource use, one can quantize the model (e.g. to 8-bit or 4-bit).  The open-source JEV models on HF likely support quantization (given many transformers do).  Even at 4-bit, a model like OpenJev (4B) might fit on one GPU.  TypeSafe has not publicly discussed quantisation, but closed models may use mixed precision under the hood.  

- **Memory/Hardware**: The footprint depends on the model size (unspecified for Jev).  A safe assumption is *several* GB of GPU RAM for inference.  OpenJev’s 4B model requires ~8–16 GB VRAM.  If using TypeSafe’s cloud, this is abstracted away.  

- **Throughput Scaling**: For high throughput, run multiple concurrent instances.  Because Jev calls are stateless (no conversation history), horizontal scaling is straightforward.  The cloud API may auto-scale, or self-hosters can use Kubernetes/containers.  

- **Failure Modes**: Since Jev cannot hallucinate, its main failures are when the state or question is beyond its training scope: e.g. if the correct answer is not listed, Jev will output the best among the given options (potentially confidently wrong).  It will always return something, so your code should check *confidence*. Low confidence (near 0.5 on a Noul, or flat probability) indicates the model is uncertain.  You should treat these as “I don’t know” signals.  There’s also a small risk of systematic biases in the training data or missing edge cases.  Continuous monitoring and fallback logic (e.g. default to human review) is recommended when confidence < a threshold.  

## Safety, Alignment, and Guardrails

- **No Hallucination by Design**: Because all outputs are pre-defined, Jev can’t produce content out of thin air.  This inherently mitigates a large class of LLM risks (hallucinated facts, undesired content).  In effect, *safety is built into the interface*.  

- **Aligned for Tasks**: Jev still needs aligned objectives: if you ask “Which queue?” and include a toxic or disallowed option, it could pick it.  Therefore, *guardrail your questions and criteria carefully*.  For example, do not include offensive categories, and use the `other` option instead of blanket “yes” in sensitive queries.  

- **Adversarial Robustness**: Since Jev’s training likely did not anticipate maliciously crafted states (e.g. input strings with hidden characters), adversaries might try to fool it.  Standard input sanitization (remove code, control chars, prompt injection) applies.  

- **Monitoring and Fallback**: Always use the confidence output to gauge trust.  A properly calibrated model can alert you when it’s unsure.  For example, only act on an answer if `confidence > 0.8`; otherwise, escalate.  This mimics human-in-the-loop patterns and avoids over-automation when Jev “isn’t sure.”  

- **Comparison with LLM Safety**: Note that Jev does **not** escape the question format.  For instance, you cannot ask an open-ended jailbreak; only questions with predefined answers.  In that sense, Jev is trivially secure against prompt injection *for outputs*.  However, it may still be susceptible if the *state* includes malicious content that tricks the model’s scoring logic; guard and filter state inputs as usual.

## Legal, IP and Licensing

- **TypeSafe Jev**: This model and name are proprietary.  TypeSafe’s blog and docs make no claim of open licensing.  Using their API or SDK will be governed by their Terms of Service and any pricing plan (e.g. token usage fees).  The “TypeSafe” name and “Jev” may be trademarked by the company.  No public patent has been filed as of this writing, but the novel training method (RLCD) could be patentable.  Caution: do not claim anything like “Jev” is open source.

- **OpenJev and Others**: The Hugging Face *OpenJev* project is MIT-licensed, so you can use and modify it freely (subject to Qwen’s license).  Qwen3.5 itself is a commercial Chinese model (non-open), so technically its weight license should be checked.  If you train your own model, ensure any base model and dataset comply with licenses (most likely CC-BY or similar for public data, proprietary license for the base).  

- **API Skill Licensing**: The TypeSafe SDK and Claude Skill are under permissive licenses (TypeSafe’s GitHub shows MIT), so integrating those into your code is allowed.  But calling the API is a paid service.  

- **Data Privacy**: If using the Jev API, note TypeSafe states “Zero data retention”.  If handling sensitive data, verify compliance (HIPAA, GDPR, etc.).  Self-hosting with OpenJev means data stays on your hardware, which may be preferable for privacy.

## Further Reading and References

- **TypeSafe Official Sources**: The primary info comes from TypeSafe’s blog *“Introducing System One Models & Jev”* and their docs (e.g. *Introduction* and *Primitives*).  These define the concept and usage.  

- **TechCrunch Article**: Fernholz’s TechCrunch interview with TypeSafe CEO provides high-level perspective (non-technical) and cites some early benchmarks.  

- **Community Analyses**: The Akka blog *“Fast, Cheap Agent Decisions”* dissects the TypeSafe claims, provides a performance/cost table, and gives examples of using Jev for routing and tool selection. Sean Goedecke’s blog on System One programming (Zero Guardrails/GitHub) offers practical tips (multi-goal hierarchies, tournament selection).  

- **Open Source Models**: HuggingFace’s [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) is a Qwen3.5-based open implementation.  The *openjev* GitHub and model card have instructions for local use and performance benchmarks (JevBench v1.2).

**Sources:** All factual claims above are drawn from primary sources as cited: the TypeSafe blog and docs, TechCrunch, Akka’s blog, and official examples. For further detail, consult those references directly.  

