# Introduction

In this repo, we tinker around with the [Branchformer](https://proceedings.mlr.press/v162/peng22a/peng22a.pdf) code. We try around the following things:
1. Dropout the CGMLP branch to observe the effect in CER performance for the Chinese Mandarin ASR dataset - [Aishell](https://www.openslr.org/33/).
2. We implement [BigBird](https://huggingface.co/blog/big-bird) based linearized attention to observe the effects.

We train all our models for $20$ epochs, with all other hyperparameters same as reported in the Branchformer paper.

# CGMLP Dropout

We observe a fatal flaw in the original Branchformer codebase, as described below:

Dropout works out by randomly (with probability $p$) zero-ing out outputs of neurons, during training. To ensure that expected output remains the same as actual outcome, at test time, the output of each neuron is multiplied by the keep-probability ($1 - p$). The original Branchformer implementation employed dropout on the Attention branch during training, however, to ensure that expected output remains same, the test time output isn't scaled by the keep-probability. We remedy that in our code implementation. For fair comparison, we retrain the attention dropout models too.

We experiment with the following settings:

- [x] cgmlp- $0.2$: Dropout CGMLP branch with $0.2$ probability

- [x] cgmlp- $0.4$: Dropout CGMLP branch with $0.4$ probability

- [x] attn- $0.2$: Dropout Attention branch with $0.2$ probability

- [x] attn- $0.4$: Dropout Attention branch with $0.4$ probability

Figure 1 highlights the trend of Character Error Rate (CER) observed while training, on the validation (_dev_) set. Table 1 reports the results obtained from the best model.

<p align="center">
  <img src="CGMLP vs Attn Dropout - CER vs Epoch.png"><br>
  <b>Figure 1</b>: <em>Chararter Error Rate (CER) trend for the validation (_dev_) set during training.</em>
</p>

<p align="center">
  <table>
    <tr>
      <th>Dropped Branch</th>
      <th>Dropout Rate</th>
      <th>Original Model</th>
      <th>Pruned Model</th>
    </tr>
    <tr>
      <td>Attention</td>
      <td>$0.2$</td>
      <td>$4.45$</td>
      <td>$5.15$</td>
    </tr>
    <tr>
      <td>Attention</td>
      <td>$0.4$</td>
      <td>$4.52$</td>
      <td>$5.02$</td>
    </tr>
    <tr>
      <td>CGMLP</td>
      <td>$0.2$</td>
      <td>$4.79$</td>
      <td>$5.35$</td>
    </tr>
    <tr>
      <td>CGMLP</td>
      <td>$0.4$</td>
      <td>$4.91$</td>
      <td>$5.27$</td>
    </tr>
  </table>
  <b>Table 1</b>: <em>Chararter Error Rate (CER) results obtained on the validation (_dev_) set, from the best trained model.</em>
</p>

# BigBird Attention

We employ the BigBird based random attention which has a linear space requirement -- $\mathcal{O}(N)$, $N$ $\rightarrow$ sequence length. Figure 2 highlights the Character Error Rate (CER) trend. We obtain a CER of $4.97$ using the best model on the validation (_dev_) set, compared to $4.22$ obtained with Fastformer based attention.

<p align="center">
<img src="BigBird - CER vs Epoch.png"></img><br>
<b>Figure 2</b>: <em>Chararter Error Rate (CER) trend for the validation (_dev_) set during training for BigBird Attention based implementation.</em>
</p>

# Credits

Link to original project on github - [ESPNET](https://github.com/espnet/espnet).
