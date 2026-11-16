# Day 2 — Deep Learning + SQL (Mon, 16 Nov)

> **Morning:** Deep Learning (Part A) + read [mindscope-cnn-project.md](mindscope-cnn-project.md)
> **Afternoon/Evening:** SQL (Part B). **Write every query by hand on paper.** Walk-ins often hand you a pen and paper.

---

# PART A — DEEP LEARNING

**How to study this part:** Read each explanation, then try saying the idea in 2–3 sentences without looking. Focus on what happens to the input, how the model learns, and why we choose each component. The formulas support the intuition; you do not need to derive them for a basic explanation.

## A1. Neural Network in plain words

A neural network is layers of small calculators (neurons). Each neuron:
1. Takes inputs, multiplies each by a **weight**, and adds a **bias**: `z = w1x1 + w2x2 + ... + b`
2. Passes `z` through an **activation function** to add non-linearity.

```
Input layer  →  Hidden layer(s)  →  Output layer
(features)      (learn patterns)     (prediction)
```

**Why activation functions?** Without them, stacking 100 layers is still just one big linear equation, so the network couldn't learn curves or complex patterns.

**Understand it:** A neuron is a learned calculation, not a hand-written rule such as "this shape means tumor." During training, weights change so that useful patterns produce useful signals. A bias shifts when a neuron activates, even when the inputs stay the same. In image models, early layers may detect simple patterns such as edges, and later layers combine them into more task-specific features.

**What makes it deep?** Multiple hidden layers let the model build representations step by step. In a dense layer, every output neuron connects to every input from the previous layer. A network's architecture decides how information flows; training decides the values of its weights. More layers increase capacity, but do not automatically improve results, especially with little data.

## A2. Activation Functions

| Function | Output range | Where used | Note |
|---|---|---|---|
| **ReLU** `max(0, x)` | 0 to ∞ | Hidden layers (common choice) | Fast, helps gradient flow for positive inputs. Downside: "dying ReLU" |
| **Leaky ReLU** | small slope for x<0 | Hidden layers | Helps reduce dying ReLU |
| **Sigmoid** | 0 to 1 | **Binary classification output** | Vanishing gradient in deep nets *(used in Mind-Scope output)* |
| **Softmax** | probabilities summing to 1 | **Multi-class output** | e.g. cat/dog/horse |
| **Tanh** | −1 to 1 | RNNs, older nets | Zero-centered |
| **GELU** | ~ReLU, smooth | **Transformers** (BERT, GPT) | |

**Understand it:** ReLU turns -3 into 0 and leaves 3 as 3. Its slope is 1 on the positive side, which helps gradients pass backward without repeatedly shrinking there. A neuron that receives only negative inputs can stop updating through ReLU; Leaky ReLU keeps a small negative-side slope. Sigmoid and tanh can saturate at extreme inputs, where their slopes become very small and learning slows.

**Choose the output for the task:** One sigmoid output works for binary classification. Softmax works when exactly one class is correct, such as cat **or** dog **or** horse. For multi-label classification, such as an image containing both a dog and a car, use separate sigmoid outputs because labels can be present independently. Regression often uses a linear output with no range restriction.

## A3. How a network learns (training loop)

1. **Forward pass:** input goes through the layers → prediction.
2. **Loss:** measure how wrong the prediction is.
   - Binary classification → **Binary Cross-Entropy** *(Mind-Scope used this)*
   - Multi-class → **Categorical Cross-Entropy**
   - Regression → **MSE / MAE**
3. **Backpropagation:** use the **chain rule** to compute how much each weight contributed to the error (the **gradient**).
4. **Gradient descent:** nudge each weight a little in the direction that reduces loss.
   `w = w − learning_rate × gradient`
5. Repeat for many **batches** and **epochs**.

**Vocabulary:**
- **Epoch** = one full pass over the whole training data.
- **Batch size** = how many samples are processed before one weight update (Mind-Scope: 32).
- **Iteration/step** = one batch update. (1445 train images / 32 = ~46 steps per epoch, which you can see in your notebook output!)
- **Learning rate** = step size. Too high → loss jumps around and diverges. Too low → painfully slow.

**Understand backprop:** A gradient measures how sensitive the current loss is to a small change in a weight. Backpropagation calculates these sensitivities efficiently, working backward through the operations using the chain rule. The optimizer then uses them to update weights. Backprop calculates the gradients; the optimizer applies the learning rule.

**Tiny example:** If a weight is 0.5, its gradient is +2, and the learning rate is 0.01, the update gives `0.5 - 0.01 × 2 = 0.48`. A positive gradient means increasing that weight would locally increase loss, so we move the other way. A small step aims to reduce loss, though noisy mini-batches mean loss does not have to fall after every update.

**Connect the terms:** With 1,445 images and batch size 32, an epoch takes 46 updates if we include the final partial batch. Ten epochs means seeing the training dataset ten times, usually with reshuffling. For binary cross-entropy, predicting 0.99 for a true negative is penalized far more than predicting 0.6: confident mistakes are expensive. Loss is the training objective; accuracy is a separate metric calculated after choosing classes.

## A4. Optimizers
- **SGD**: basic gradient descent on mini-batches.
- **SGD + Momentum**: keeps "speed" from previous steps, so it rolls past small bumps.
- **Adam** ⭐: momentum + an adaptive learning rate per weight. **The default choice.** *(Mind-Scope used Adam.)*
- **AdamW**: Adam with proper weight decay, the standard for training transformers.

**Understand it:** SGD uses the current batch's gradient. Momentum carries some of the previous update direction forward, helping smooth zigzags. Adam also tracks gradient magnitudes, adapting update sizes for different parameters. It is a useful starting point, but it still needs a sensible learning rate and validation. AdamW applies weight decay separately from Adam's adaptive gradient update, encouraging smaller weights.

**Learning-rate scheduling:** A scheduler changes the learning rate during training. Larger early steps can help make progress, while smaller later steps help refine the solution. A falling training loss alone does not prove the chosen optimizer is better; compare validation performance and training cost too.

## A5. Common problems & fixes

| Problem | What happens | Fix |
|---|---|---|
| **Vanishing gradient** | Gradients become tiny in early layers, so they stop learning | ReLU, BatchNorm, residual connections (ResNet), good init |
| **Exploding gradient** | Gradients become huge; updates may destabilize training | Gradient clipping, suitable initialization, lower LR for unstable updates |
| **Overfitting** | Train ≫ validation | Dropout, augmentation, L2, early stopping, more data |
| **Slow training** | | Adam, BatchNorm, LR schedulers, GPU |

- **Dropout:** randomly turn off e.g. 30% of neurons each training step, so the network can't rely on any single neuron. (Turned off at inference.)
- **Batch Normalization:** normalizes each layer's outputs per batch (mean 0, std 1, then learnable scale and shift). Training is faster and more stable. *(Mind-Scope has a BN layer.)*
- **Early stopping:** watch validation loss and stop when it stops improving.

**Why gradients vanish or explode:** Backprop multiplies derivatives across layers or sequence steps. Repeatedly multiplying small values can make early-layer gradients tiny; multiplying large values can make them huge. Residual connections provide a more direct path for information and gradients. Gradient clipping limits gradient size before an update, while good initialization helps keep signals at useful scales.

**Dropout vs BatchNorm:** Dropout is regularization: temporarily removing random activations discourages the network from depending too heavily on one pathway. BatchNorm normalizes intermediate activations using batch statistics during training, followed by a learned scale and shift. At inference, dropout is disabled and BatchNorm normally uses stored running statistics. BatchNorm can be unreliable with very small batches; it is not a substitute for input preprocessing.

**Read the learning curves:** Falling training loss with rising validation loss suggests overfitting. High loss on both sets suggests underfitting or a training/data issue. Early stopping usually allows several epochs without improvement (**patience**) and restores the best validation checkpoint, rather than keeping the final weights. Data augmentation creates varied training examples, but transformations must preserve the label and make sense for the domain.

## A6. CNN — Convolutional Neural Networks (images)

Why not a normal dense network for images? A 240×240×3 image = 172,800 inputs. Fully connecting that is huge, and it ignores the fact that nearby pixels are related.

**Building blocks:**
1. **Convolution layer:** small **filters/kernels** (e.g. 3×3, 7×7) slide across the image and detect patterns. Early layers find edges, deeper layers find shapes and then objects. The **same filter is reused everywhere** (weight sharing), so you need far fewer parameters.
   - **Stride**: how many pixels the filter moves each step.
   - **Padding**: add zeros around the border to control the output size.
   - **Output size formula** ⭐: `floor((N − F + 2P) / S) + 1` for each spatial dimension, assuming dilation 1 (N = input, F = filter, P = padding per side, S = stride).
2. **Activation (ReLU)** after each convolution.
3. **Pooling layer:** shrinks the image. **Max pooling** keeps the max value in each window. That means fewer computations and some position tolerance.
4. **Flatten → Dense layers → Output** (sigmoid / softmax).

**Understand convolution:** A filter multiplies a small patch by learned weights and sums the results to produce one activation. Sliding the same filter across the image creates a **feature map**, showing where that pattern occurs. A typical filter spans all input channels; using 32 filters produces 32 output channels. The weights are learned during training, rather than us manually specifying an edge detector.

**A small shape example:** A `32×32×3` image passed through 16 filters of size `3×3`, stride 1, and no padding becomes `30×30×16`. A `2×2` max-pooling layer with stride 2 then makes it `15×15×16`. Pooling shrinks height and width, while the channel count stays the same. A max-pooling window containing `[1, 3; 2, 0]` keeps 3, preserving the strongest local activation while discarding some detail.

**Flatten vs global average pooling:** Flatten turns every spatial activation into one long vector; a following dense layer can therefore have many weights. Global average pooling averages each feature map into one number, giving one value per channel. It often makes the classification head smaller, although it discards spatial detail that some tasks need.

**Famous architectures (name-drop):** LeNet → AlexNet → **VGG16** (simple stacked 3×3 convs) → **ResNet** (skip connections let you train 100+ layers) → **EfficientNet** → **Vision Transformers (ViT)**.

**Transfer learning** ⭐: take a model pre-trained on ImageNet (millions of images), freeze the early layers, and retrain only the top on your small dataset. This is the go-to when you have little data. (This is what you'd do to improve Mind-Scope.)

**Understand the two stages:** First, use the pretrained backbone as a feature extractor: freeze it and train a new output head for your labels. Then, if validation results justify it, unfreeze some later layers and fine-tune with a smaller learning rate. Starting from useful visual features can reduce the data and training needed. It is still an experiment: natural-image features may transfer imperfectly to medical scans, and patient-level splitting remains essential.

**Why ResNet helps:** A residual block adds its input to a learned transformation, roughly `output = x + F(x)` when shapes match. The block can learn a correction instead of rebuilding the whole representation, and the shortcut helps gradients travel through deep networks. ViT is an alternative image architecture based on attention over image patches, rather than a conventional CNN.

➡️ **Now read [mindscope-cnn-project.md](mindscope-cnn-project.md)**. It applies all of this to your own project with real numbers.

## A7. RNN / LSTM (quick)
- **RNN**: processes sequences step by step and keeps a "memory" (hidden state). Problem: forgets long-range context because of vanishing gradients.
- **LSTM / GRU**: add learned **gates** to control what to retain and update. LSTMs have forget/input/output gates; GRUs use reset/update gates.
- **Why transformers became dominant in language modeling:** recurrent time steps are sequential, while attention allows much more parallel work during training and directly connects distant tokens. RNNs remain useful for some streaming and time-series tasks.

**Understand it:** An RNN reads a sequence one item at a time and updates a hidden state that summarizes earlier inputs. In "The movie was not very good," the model needs earlier words such as "not" to interpret "good." LSTM adds a cell state with gated updates, helping preserve information over longer stretches. Gates are learned numerical controls, not hand-written decisions about which words matter.

## A8. Transformers & Attention ⭐ (GenAI role, so expect this)

**Core idea, self-attention:** every word looks at every other word in the sentence and decides how much to "pay attention" to it.
> "The **animal** didn't cross the street because **it** was tired." → attention helps "it" connect to "animal".

**How (high level):**
- Each token becomes 3 vectors: **Query** (what am I looking for?), **Key** (what do I contain?), **Value** (what info do I pass on?).
- `Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`
  - Q·Kᵀ = similarity scores between tokens.
  - ÷ √d keeps the numbers stable.
  - softmax turns them into weights, and the weighted sum of the Values is the output.
- **Multi-head attention**: several attention "heads" run in parallel, each learning different relationships (grammar, meaning, position...).
- **Positional encoding**: attention has no sense of order by itself, so position info is added.
- Each block = attention + feed-forward + residual connections + layer norm, stacked N times.

**Understand Q, K, V:** These vectors are learned projections of token representations. For the token "it," its query is compared with the keys of allowed context tokens to produce relevance scores. Those scores weight the corresponding value vectors, creating a context-aware representation of "it." The animal example illustrates a possible relationship; individual heads are not guaranteed to correspond neatly to human concepts.

**What the rest of the block does:** Attention mixes information across token positions; the feed-forward network transforms the representation at each position. Residual connections help information and gradients flow, while layer normalization stabilizes activations using statistics within each token's representation rather than across the batch. Position information lets the model distinguish sequences such as "dog bites man" and "man bites dog."

**BERT vs GPT:** A BERT-style encoder can use tokens on both sides to build a representation. A GPT-style decoder uses a **causal mask** so each position attends only to itself and earlier positions, supporting next-token prediction without seeing the answer. During training, many positions can be processed in parallel under that mask; during ordinary generation, new tokens are produced sequentially. Standard full attention also becomes expensive as sequence length grows because it forms pairwise token scores.

**Three families:**
| Type | Example | Good at |
|---|---|---|
| **Encoder-only** | BERT | Understanding: classification, **embeddings**, NER |
| **Decoder-only** | GPT, Llama, Claude | **Generation** (predicts the next token) |
| **Encoder-Decoder** | T5, BART | Translation, summarization |

**LLM basics to know:**
- **Token**: a piece of a word (~4 characters in English). Cost and context limits are counted in tokens.
- **Context window**: the max tokens the model can see at once.
- **Temperature**: adjusts how concentrated the next-token probabilities are. Lower values favor likely tokens; higher values allow more variety. Low temperature does **not** guarantee factual answers or perfectly repeatable runs.
- **Top-p (nucleus sampling)**: sample only from the smallest set of tokens whose probabilities add up to p.
- **Hallucination**: a plausible but unsupported or false answer. Mitigations include reliable retrieved context, grounding instructions, abstention, and checking claims and citations. These reduce risk; none guarantees correctness.

**Understand generation:** The model repeatedly predicts a distribution over the next token, selects one, appends it, and continues. A token can be a word, part of a word, punctuation, or another text unit; the characters-per-token estimate varies by language and content. The context window is a capacity limit, not a promise that the model uses every detail reliably. A fluent response can be wrong because plausible continuation and verified truth are different objectives.

## A9. Prompting vs RAG vs Fine-tuning ⭐⭐ (classic GenAI DS question)

| | **Prompt Engineering** | **RAG** | **Fine-tuning** |
|---|---|---|---|
| What it changes | The instructions | The **knowledge** the model sees | The model's **behavior/weights** |
| Use when | Task is general, model already knows how | Need **private / fresh / changing data**, need citations | Need a specific **style, format, tone, domain language**, or a smaller cheaper model for one narrow task |
| Main effort / cost | Prompt design and inference | Retrieval, indexing, and inference | Training data, training, evaluation, and inference |
| Updating knowledge | Edit the prompt | **Just re-index the documents** | Retrain |
| Factual reliability | Depends on model and supplied context | Can improve grounding if retrieval and source use are good | Does not guarantee factual correctness |

**Answer pattern:** *"I always start with prompting. If the model lacks knowledge (company docs, recent data), I add RAG. I'd only fine-tune if we need consistent behavior or format that prompting can't achieve, or to distill into a smaller, cheaper model. Often the best is RAG + a light fine-tune."*

**Rule of thumb:** RAG = **knowledge** problem. Fine-tuning = **behavior** problem.

**A concrete example:** For an assistant answering company leave-policy questions, start with clear instructions. If it needs the current policy, retrieve the relevant policy passages and include them in its context: that is RAG, and it does not change model weights. If it repeatedly fails to follow a specialized response style despite good instructions and examples, fine-tuning may help. Fine-tuning can also teach knowledge, but changing documents is usually easier to maintain through retrieval. The methods can be combined, and actual costs depend on workload and model size.

## A10. Fine-tuning methods
- **Full fine-tuning**: update all the weights. Very expensive (needs lots of GPU memory).
- **PEFT (Parameter-Efficient Fine-Tuning)**: update only a tiny fraction of the parameters.
  - **LoRA** ⭐: freeze the original weights and add small **low-rank matrices** (A×B) beside them. Only those small matrices are trained (often <1% of the params). You can swap LoRA adapters for different tasks.
  - **QLoRA**: LoRA + the base model loaded in **4-bit quantized** form, so you can fine-tune large models on a single GPU.
- **Instruction tuning / SFT**: train on (instruction, ideal response) pairs.
- **RLHF / DPO**: align the model with human preferences (which answer is better).
- **Quantization**: represent weights, and sometimes activations, with lower precision (e.g. FP16 → INT8/INT4). This reduces memory; speed and quality effects depend on the model, method, and hardware.
- **Distillation**: a small "student" model learns to mimic a large "teacher".

**Understand LoRA:** Instead of updating a large weight matrix directly, keep it frozen and learn a smaller correction represented by two low-rank matrices. This reduces trainable parameters and optimizer memory. The model still needs computation through its frozen base, so fewer trainable parameters does not make training free. QLoRA also stores the frozen base in 4-bit form; whether it fits on one GPU depends on model size, sequence length, and available memory.

**Separate the ideas:** SFT describes training on desired examples; LoRA describes which parameters you update, so you can perform SFT using LoRA. RLHF commonly uses a learned reward signal with reinforcement learning, while DPO directly learns from preferred versus rejected response pairs without a separate RL training loop. Quantization changes numerical precision; distillation trains a student using a teacher's outputs or other signals.

## A11. Embeddings (bridge to Day 3)
- An **embedding** = text (or an image) turned into a vector of numbers that captures meaning.
- Similar meanings end up as nearby vectors: "king" ≈ "queen", "car" ≈ "vehicle".
- Created by encoder models (e.g. `text-embedding-3-small`, `bge`, `all-MiniLM-L6-v2`).
- This is the foundation of **RAG and vector search** (Day 3).

**Understand it:** "How do I reset my password?" and "I forgot my login password" can have similar embeddings despite different wording. Search embeds the query with a compatible embedding model and compares it with stored document vectors, often using cosine similarity. The closest vectors are candidates for relevant context, not guaranteed correct answers. Quality depends on the embedding model, document chunks, and task; sharing words or a topic does not necessarily mean two passages answer the same question.

### ✅ DL Self-Test
1. Why do we need activation functions?
2. Explain backprop in 3 sentences.
3. Epoch vs batch vs iteration?
4. What does dropout do? What does BatchNorm do?
5. CNN output size: input 244, filter 7, stride 1, padding 0 → ? *(Answer: 238)*
6. Why is max pooling used?
7. What is transfer learning and when would you use it?
8. Explain self-attention in simple words. What are Q, K, V?
9. BERT vs GPT?
10. When would you choose RAG vs fine-tuning?
11. What is LoRA? QLoRA?
12. What does temperature do?

---

# PART B — SQL ⭐⭐⭐

> Practice on paper first, then verify online (e.g. sqliteonline.com, DB Fiddle, or LeetCode SQL 50).

## B1. Order of execution (explains many errors)
```
FROM / JOIN  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  DISTINCT  →  ORDER BY  →  LIMIT
```
That's why:
- You **can't use a SELECT alias in WHERE** (WHERE runs before SELECT).
- **WHERE filters rows before grouping. HAVING filters groups after aggregation.**

## B2. Practice tables (use for all examples)

```sql
employees(emp_id, name, dept_id, salary, manager_id, hire_date)
departments(dept_id, dept_name)
orders(order_id, customer_id, order_date, amount)
customers(customer_id, name, city, signup_date)
```

## B3. JOINs
| Join | Returns |
|---|---|
| **INNER JOIN** | Only matching rows from both |
| **LEFT JOIN** | All rows from the left + matches from the right (NULL if none) |
| **RIGHT JOIN** | All rows from the right + matches from the left |
| **FULL OUTER JOIN** | Everything from both (MySQL doesn't support it, so use LEFT UNION RIGHT) |
| **CROSS JOIN** | Every combination (cartesian product) |
| **SELF JOIN** | A table joined to itself (employee → manager) |

```sql
-- Employees with their department name
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- Customers who NEVER ordered (classic!)
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- Employee with their manager's name (self join)
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- Employees earning more than their manager
SELECT e.name
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

## B4. GROUP BY + HAVING + CASE WHEN
```sql
-- Avg salary per department, only departments with > 5 employees
SELECT dept_id, COUNT(*) AS emp_count, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
HAVING COUNT(*) > 5
ORDER BY avg_sal DESC;

-- Salary bands with CASE WHEN
SELECT name, salary,
       CASE
         WHEN salary >= 100000 THEN 'High'
         WHEN salary >= 50000  THEN 'Medium'
         ELSE 'Low'
       END AS band
FROM employees;

-- Conditional aggregation (pivot-style counting)
SELECT dept_id,
       SUM(CASE WHEN salary >= 100000 THEN 1 ELSE 0 END) AS high_earners,
       SUM(CASE WHEN salary <  100000 THEN 1 ELSE 0 END) AS others
FROM employees
GROUP BY dept_id;
```

## B5. Subqueries & CTEs
```sql
-- Employees earning above the company average (subquery)
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Employees earning above THEIR department's average (correlated subquery)
SELECT e.name, e.salary, e.dept_id
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id);

-- Same thing with a CTE (cleaner, readable — prefer this)
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```
**CTE = a named temporary result (`WITH ... AS`).** Makes complex queries readable and step-by-step.

## B6. Window Functions ⭐⭐⭐ (the most-asked topic)

**Key idea:** like GROUP BY, but it **does NOT collapse rows**. Every row stays, and it gets an extra calculated column.

```sql
function() OVER (PARTITION BY <group> ORDER BY <col>)
```
- `PARTITION BY` = "restart the calculation for each group"
- `ORDER BY` = "in what order to rank or accumulate"

### ROW_NUMBER vs RANK vs DENSE_RANK (guaranteed question)
Salaries: 100, 90, 90, 80

| salary | ROW_NUMBER | RANK | DENSE_RANK |
|---|---|---|---|
| 100 | 1 | 1 | 1 |
| 90 | 2 | 2 | 2 |
| 90 | 3 | 2 | 2 |
| 80 | 4 | **4** (skips 3) | **3** (no gap) |

- **ROW_NUMBER**: always unique (1, 2, 3, 4), even for ties.
- **RANK**: ties share a rank, then it **skips**.
- **DENSE_RANK**: ties share a rank, **no gap**. ← use this for "Nth highest salary".

### Classic Problem 1: 2nd highest salary (overall)
```sql
-- Method 1: window function (best)
SELECT DISTINCT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk = 2;

-- Method 2: subquery
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: LIMIT/OFFSET (MySQL)
SELECT DISTINCT salary FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

### Classic Problem 2: Highest (or top 3) salary in EACH department
```sql
WITH ranked AS (
    SELECT name, dept_id, salary,
           DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
    FROM employees
)
SELECT name, dept_id, salary
FROM ranked
WHERE rnk <= 3;      -- rnk = 1 for top only, rnk = 2 for 2nd highest per dept
```

### Classic Problem 3: Find and delete duplicates
```sql
-- Find duplicate emails
SELECT email, COUNT(*)
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Keep one row per email, remove the rest
WITH d AS (
    SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
    FROM users
)
DELETE FROM users WHERE id IN (SELECT id FROM d WHERE rn > 1);
-- (MySQL needs this as a JOIN-based delete or a derived table; the idea is what matters)
```

### Classic Problem 4: Running total & rolling 7-day average
```sql
SELECT order_date, daily_sales,
       SUM(daily_sales) OVER (ORDER BY order_date) AS running_total,
       AVG(daily_sales) OVER (
           ORDER BY order_date
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7day_avg
FROM (
    SELECT order_date, SUM(amount) AS daily_sales
    FROM orders
    GROUP BY order_date
) t;
```

### Classic Problem 5: LAG / LEAD (compare with the previous row)
```sql
-- Month-over-month revenue growth
WITH monthly AS (
    SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(amount) AS revenue
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 2) AS growth_pct
FROM monthly;
```
- `LAG(col, n)` = value from n rows **before**. `LEAD(col, n)` = value from n rows **after**.

### Classic Problem 6: Each customer's first order
```sql
SELECT *
FROM (
    SELECT o.*, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn
    FROM orders o
) t
WHERE rn = 1;
```

### Classic Problem 7: % share of each department's salary in the total
```sql
SELECT dept_id,
       SUM(salary) AS dept_total,
       ROUND(100.0 * SUM(salary) / SUM(SUM(salary)) OVER (), 2) AS pct_of_company
FROM employees
GROUP BY dept_id;
```

### Classic Problem 8: Consecutive days (users active 3 days in a row)
```sql
-- Trick: date minus row_number is constant for consecutive dates
WITH t AS (
    SELECT user_id, login_date,
           DATE_SUB(login_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) DAY) AS grp
    FROM (SELECT DISTINCT user_id, login_date FROM logins) x
)
SELECT user_id
FROM t
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

## B7. Quick theory questions
- **DELETE vs TRUNCATE vs DROP:** DELETE removes rows (can use WHERE, can roll back). TRUNCATE removes all rows quickly (no WHERE). DROP removes the whole table.
- **WHERE vs HAVING:** rows before grouping vs groups after aggregation.
- **UNION vs UNION ALL:** UNION removes duplicates (slower). UNION ALL keeps all.
- **Primary key vs Foreign key:** unique row ID vs a reference to another table's PK.
- **Index:** like a book's index. Faster reads, slightly slower writes. Use on columns in WHERE/JOIN/ORDER BY.
- **Normalization:** split data into tables to avoid duplication (1NF, 2NF, 3NF). **Denormalization** for faster analytics reads.
- **COUNT(*) vs COUNT(col):** COUNT(col) ignores NULLs.
- **NULL handling:** `IS NULL`, not `= NULL`. `COALESCE(col, 0)` replaces NULL.
- **ACID:** Atomicity, Consistency, Isolation, Durability (transactions).
- **SQL vs NoSQL:** SQL = structured, relations, joins (MySQL, Postgres). NoSQL = flexible schema, scale (MongoDB, which you've used!).
- **pgvector:** a Postgres extension for storing embeddings and doing vector similarity search in SQL. A nice bridge between SQL and RAG.

## B8. Pandas equivalents (they may ask "do this in pandas")
```python
df.groupby("dept_id")["salary"].mean()                         # GROUP BY
df[df.salary > 50000]                                           # WHERE
pd.merge(emp, dept, on="dept_id", how="left")                   # LEFT JOIN
df["rnk"] = df.groupby("dept_id")["salary"].rank(method="dense", ascending=False)  # DENSE_RANK
df["prev"] = df["revenue"].shift(1)                             # LAG
df["roll7"] = df["sales"].rolling(7).mean()                     # rolling avg
df.drop_duplicates(subset="email", keep="first")                # dedupe
df.isna().sum(); df.fillna(0)                                   # nulls
```

### ✅ SQL Self-Test (write on paper, no peeking)
1. 2nd highest salary, three ways.
2. Top 2 earners per department.
3. Customers with no orders.
4. Departments with average salary > 60,000.
5. Find duplicate emails, and delete them keeping one.
6. Running total of daily sales.
7. 7-day rolling average.
8. Month-over-month growth % with LAG.
9. Employees earning more than their manager.
10. RANK vs DENSE_RANK vs ROW_NUMBER with an example.
