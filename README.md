# Ex.No.1 – Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)

**Name:** Harish K
**Register No:** 212225230094
**Department:** AI&DS

---

## Aim
To compare the behaviour of different Generative AI tools (**Gemini** and **Perplexity**) using identical prompts, and to develop a comprehensive report on the fundamentals of Generative AI and Large Language Models (LLMs).

## Experiment
Develop a comprehensive report for the following exercises:

1. Explain the foundational concepts of Generative AI.
2. Focusing on Generative AI architectures (like Transformers).
3. Generative AI architectures and their applications.
4. Generative AI impact of scaling in LLMs.
5. Explain about LLMs and how they are built.

## Algorithm

**Step 1: Define Scope and Objectives**
- 1.1 Identify the goal of the report (educational, research, tech overview)
- 1.2 Set the target audience level (students, professionals)
- 1.3 Draft a list of core topics to cover

**Step 2: Create Report Skeleton/Structure**
- 2.1 Title Page
- 2.2 Abstract or Executive Summary
- 2.3 Table of Contents
- 2.4 Introduction
- 2.5 Main Body Sections: Introduction to AI and Machine Learning, What is Generative AI?, Types of Generative AI Models (GANs, VAEs, Diffusion Models), Introduction to LLMs, Architecture of LLMs (Transformer, GPT, BERT), Training Process and Data Requirements, Use Cases and Applications, Limitations and Ethical Considerations, Future Trends
- 2.6 Conclusion
- 2.7 References

**Step 3: Research and Data Collection**
- 3.1 Gather recent academic papers, blog posts, and official docs
- 3.2 Extract definitions, explanations, diagrams, and examples
- 3.3 Cite all sources properly

**Step 4: Content Development**
- 4.1 Write each section in clear, simple language
- 4.2 Include diagrams, figures, and charts where needed
- 4.3 Highlight important terms and definitions
- 4.4 Use examples and real-world analogies

**Step 5: Visual and Technical Enhancement**
- 5.1 Add tables and comparison charts
- 5.2 Use tools like Canva, PowerPoint, or LaTeX for formatting
- 5.3 Add code snippets or pseudocode for LLM working (optional)

**Step 6: Review and Edit**
- 6.1 Proofread for grammar, spelling, and clarity
- 6.2 Ensure logical flow and consistency
- 6.3 Validate technical accuracy
- 6.4 Peer-review or use tools like Grammarly or ChatGPT for suggestions

**Step 7: Finalize and Export**
- 7.1 Format the report professionally
- 7.2 Export as PDF or desired format
- 7.3 Prepare a brief presentation if required (optional)

---

## Output

The same five prompts were given to **Gemini** and **Perplexity**, each in a fresh chat.

### Prompt 1: Foundational Concepts of Generative AI
> Explain the foundational concepts of Generative AI for a second-year B.Tech AI & Data Science student. Cover: what it is, how it differs from discriminative AI, main model families (GANs, VAEs, diffusion models, autoregressive models), and training vs. inference. Use simple language, one real-world analogy, and a short summary table. Keep it under 500 words.

<details>
<summary><b>Gemini Output</b></summary>

The Transformer architecture, which revolutionized Generative AI by processing entire sequences of data simultaneously rather than word-by-word, was introduced in the 2017 paper **"Attention Is All You Need"** by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

**Self-Attention**
Instead of reading sequentially, self-attention allows the model to look at the entire sequence at once and mathematically score how strongly every word relates to every other word. This allows the model to dynamically resolve context—for example, determining whether "bank" refers to a river edge or a financial institution based on surrounding words.

**Multi-Head Attention**
Rather than computing self-attention just once, the Transformer runs multiple attention mechanisms (or "heads") in parallel. This allows the network to track different types of relationships simultaneously—one head might focus on grammatical structure, while another tracks pronoun references or emotional tone.

**Positional Encoding**
Because the model processes all words simultaneously, it natively has no concept of word order. To fix this, positional encodings—fixed mathematical signals like sine and cosine waves—are added to the input data, injecting a sense of sequence and distance into the model's calculations.

**Encoder vs. Decoder**
The original architecture consists of two halves:

* **Encoder:** Reads the entire input sequence bidirectionally to build a deep, contextual mathematical representation of the text.
* **Decoder:** Generates the output sequentially. It uses a "masked" self-attention mechanism, meaning it is only mathematically allowed to look at previous words—never future words—to predict the next token. It also features a cross-attention layer that connects back to the Encoder's output.

**GPT vs. BERT**
Following the original paper, researchers split the architecture to specialize models for different tasks:

* **BERT (Bidirectional Encoder Representations from Transformers):** Built exclusively from **Encoder** blocks. Because it reads text in both directions simultaneously, it possesses deep contextual understanding, making it the standard for classification, search, and sentiment analysis. It does not generate text.
* **GPT (Generative Pre-trained Transformer):** Built exclusively from **Decoder** blocks. It reads unidirectionally (left-to-right) to predict the next word. This constraint makes it exceptional at open-ended text generation, chat, and summarization.

**Architecture Diagram**

```text
       Input Text
           ↓
 [ Embeddings + Positional Encoding ]
           ↓
  ==================================
  |        ENCODER BLOCK           |  (BERT uses this half)
  | 1. Multi-Head Self-Attention   |
  | 2. Feed-Forward Network        |
  ==================================
           ↓ (Context Vectors)
  ==================================
  |        DECODER BLOCK           |  (GPT uses this half)
  | 1. Masked Self-Attention       |
  | 2. Encoder-Decoder Attention   |
  | 3. Feed-Forward Network        |
  ==================================
           ↓
   [ Linear Layer & Softmax ]
           ↓
      Output Probabilities

```

[Understanding Transformers, Self-Attention, and GPT vs. BERT](https://www.youtube.com/watch?v=Gy84sFW5Uv8)
This video provides a clear visual breakdown of how self-attention works and how the original architecture was split to create the specialized BERT and GPT models.
</details>

<details>
<summary><b>Perplexity Output</b></summary>

Generative AI architectures are neural networks that learn patterns from data and create new outputs—text, images, audio, or code. The **Transformer**, introduced in 2017, is the dominant architecture for modern language models because it processes tokens in parallel and captures long-range context through attention. [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Transformer architecture

```text
                Original Transformer
                ┌─────────────────────────────┐
Input tokens →  │ Embedding + Positional Encoding │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │                 ENCODER                     │
        │  Multi-Head Self-Attention                  │
        │  Add & Normalize                            │
        │  Feed-Forward Network                       │
        │  Add & Normalize        × N layers          │
        └──────────────────────┬──────────────────────┘
                               │ encoder representations
        ┌──────────────────────▼──────────────────────┐
        │                 DECODER                     │
        │  Masked Multi-Head Self-Attention           │
        │  Cross-Attention (decoder → encoder)        │
        │  Add & Normalize                            │
        │  Feed-Forward Network                       │
        │  Add & Normalize        × N layers          │
        └──────────────────────┬──────────────────────┘
                               │
                         Linear + Softmax
                               │
                       Output probabilities
```

The original paper is **“Attention Is All You Need” (2017)** by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Self-attention

Self-attention lets every token examine every other token in the same sequence and decide which ones matter for its representation. Each token is projected into a **query (Q)**, **key (K)**, and **value (V)** vector; attention scores are computed as:

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The softmax turns similarity scores into weights, which are used to blend the value vectors. For example, in “The animal didn’t cross the street because it was tired,” self-attention helps the model associate “it” with “animal.” [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Multi-head attention

Instead of computing one attention pattern, the model runs several attention “heads” in parallel. Each head uses different learned projections of Q, K, and V, allowing it to focus on different relationships—such as syntax, coreference, or nearby-word dependencies. Their outputs are concatenated and projected back into the model dimension. [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Positional encoding

Because attention itself has no inherent sense of word order, positional information is added to token embeddings. The original Transformer uses sinusoidal functions of different frequencies; later models, including BERT, often use learned position embeddings. This lets the model distinguish “dog bites man” from “man bites dog.” [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Encoder vs. decoder

| Component | Role | Attention used |
|---|---|---|
| Encoder | Builds a contextual representation of the input sequence | Bidirectional self-attention |
| Decoder | Generates output tokens one at a time | Masked self-attention plus cross-attention to encoder outputs |

The encoder sees the full input at once, while the decoder’s mask prevents it from looking at future tokens during generation. Cross-attention connects decoder queries with encoder keys and values, which is useful for tasks such as translation. [proceedings.neurips](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## GPT vs. BERT

**GPT** uses a Transformer **decoder** and is trained to predict the next token from left-to-right context. It is therefore naturally suited to text generation. GPT-1, *“Improving Language Understanding by Generative Pre-Training”* (2018), was created by Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. [s10251.pcdn](https://s10251.pcdn.co/pdf/2018-radford-gpt-1.pdf)

**BERT** uses a Transformer **encoder** and is trained with masked language modeling: it predicts randomly hidden words using context from both directions. This makes it strong for understanding tasks such as classification, question answering, and search relevance. BERT was introduced in *“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”* (2018) by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [arxiv](https://arxiv.org/abs/1810.04805)

In short: **GPT generates; BERT understands.** GPT is unidirectional and decoder-based, while BERT is bidirectional and encoder-based. [ai-papers.aktagon](https://ai-papers.aktagon.com/papers/gpt/)
</details>

---

### Prompt 2: Generative AI Architectures (Transformers)
> Explain Generative AI architectures with a focus on the Transformer. Describe self-attention, multi-head attention, positional encoding, encoder vs. decoder, and how GPT and BERT differ. Include a text-based diagram of the architecture and name the original paper with its year and authors. Keep it under 600 words.

<details>
<summary><b>Gemini Output</b></summary>

*(Paste Gemini's full response here)*

</details>

<details>
<summary><b>Perplexity Output</b></summary>

*(Paste Perplexity's full response here)*

</details>

---

### Prompt 3: Generative AI Architectures and Applications
> Explain how Generative AI architectures map to real applications. For each of Transformers, GANs, VAEs and Diffusion models, give two applications (text, image, audio, code, healthcare, etc.) and explain why that architecture suits each one. Present the answer as a table, then add a 100-word note on limitations.

<details>
<summary><b>Gemini Output</b></summary>

*(Paste Gemini's full response here)*

</details>

<details>
<summary><b>Perplexity Output</b></summary>

*(Paste Perplexity's full response here)*

</details>

---

### Prompt 4: Impact of Scaling in LLMs
> Explain the impact of scaling in Large Language Models. Cover parameters, training data, and compute; scaling laws; emergent abilities; the Chinchilla finding; and the downsides (cost, energy, bias, hallucination). Give concrete examples comparing GPT-3 and later models, and say clearly where you are uncertain.

<details>
<summary><b>Gemini Output</b></summary>

*(Paste Gemini's full response here)*

</details>

<details>
<summary><b>Perplexity Output</b></summary>

*(Paste Perplexity's full response here)*

</details>

---

### Prompt 5: What an LLM Is and How It Is Built
> Explain what a Large Language Model is and how it is built, step by step: data collection and cleaning, tokenization, pre-training, fine-tuning, RLHF, evaluation, and deployment. Add a short pseudocode snippet showing next-token prediction. Aim it at an engineering student, in under 600 words.

<details>
<summary><b>Gemini Output</b></summary>

*(Paste Gemini's full response here)*

</details>

<details>
<summary><b>Perplexity Output</b></summary>

*(Paste Perplexity's full response here)*

</details>

---

## Comparison of Gemini and Perplexity

Scores are out of 5 (5 = best). Speed is the time taken to complete the response.

| Criteria | Prompt | Gemini | Perplexity |
|---|---|---|---|
| **Accuracy** | P1 | | |
| | P2 | | |
| | P3 | | |
| | P4 | | |
| | P5 | | |
| **Creativity** | Overall | | |
| **Hallucination** (errors found) | Overall | | |
| **Reasoning** | Overall | | |
| **Speed** (seconds) | P1 | | |
| | P2 | | |
| | P3 | | |
| | P4 | | |
| | P5 | | |
| **Engineering Usefulness** | Overall | | |

### Observations
- **Accuracy:**
- **Creativity:**
- **Hallucination:**
- **Reasoning:**
- **Speed:**
- **Engineering usefulness:**

---

## Result
Five identical prompts covering the fundamentals of Generative AI and LLMs were given to Gemini and Perplexity. The responses were compared on accuracy, creativity, hallucination, reasoning, speed, and engineering usefulness. *(Write one or two sentences here: which tool was more accurate, which was faster, which cited sources, and which you would use for engineering work.)*

Thus, the comprehensive report on the fundamentals of Generative AI and LLMs, along with the comparison of the two AI tools, was successfully completed.
