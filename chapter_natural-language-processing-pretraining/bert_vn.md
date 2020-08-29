
<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Bidirectional Encoder Representations from Transformers (BERT)
-->

# Biểu diễn Mã hoá hai chiều từ Transformer (*Bidirectional Encoder Representations from Transformers* - BERT)
:label:`sec_bert`


<!--
We have introduced several word embedding models for natural language understanding.
After pretraining, the output can be thought of as a matrix where each row is a vector that represents a word of a predefined vocabulary.
In fact, these word embedding models are all *context-independent*.
Let us begin by illustrating this property.
-->

Chúng tôi đã giới thiệu một vài mô hình embedding từ cho bài toán hiểu ngôn ngữ tự nhiên.
Sau khi tiền huấn luyện, đầu ra có thể được coi là một ma trận trong đó mỗi hàng là một vector biểu diễn cho một từ trong bộ từ vựng đã được định nghĩa trước.
Trong thực tế, tất cả các mô hình embedding từ này đều có tính chất *độc lập ngữ cảnh* (_context-independent_).
Chúng ta sẽ bắt đầu bằng việc mô tả tính chất này.

<!--
## From Context-Independent to Context-Sensitive
-->

## Từ Độc lập Ngữ cảnh tới Nhạy Ngữ cảnh


<!--
Recall the experiments in :numref:`sec_word2vec_pretraining` and :numref:`sec_synonyms`.
For instance, word2vec and GloVe both assign the same pretrained vector to the same word regardless of the context of the word (if any).
Formally, a context-independent representation of any token $x$ is a function $f(x)$ that only takes $x$ as its input.
Given the abundance of polysemy and complex semantics in natural languages, context-independent representations have obvious limitations.
For instance, the word "crane" in contexts "a crane is flying" and "a crane driver came" has completely different meanings;
thus, the same word may be assigned different representations depending on contexts.
-->

Hãy nhớ lại các thí nghiệm trong :numref:`sec_word2vec_pretraining` và :numref:`sec_synonyms`.
Ví dụ, cả word2vec và GloVe đều gán cùng một vector được tiền huấn luyện cho cùng một từ bất kể ngữ cảnh của nó như thế nào (nếu có).
Về mặt hình thức, biểu diễn độc lập ngữ cảnh của một token bất kỳ $x$ là một hàm $f(x)$ chỉ nhận $x$ làm đầu vào.
Do hiện tượng đa nghĩa cũng như sự phức tạp ngữ nghĩa xuất hiện khá phổ biến trong ngôn ngữ tự nhiên, biểu diễn độc lập ngữ cảnh có những hạn chế rõ ràng.
Ví dụ, từ "crane" trong ngữ cảnh "a crane is flying (một con sếu đang bay)" và ngữ cảnh "a crane driver came (tài xế xe cần cẩu đã tới)" có nghĩa hoàn toàn khác nhau;
do đó, cùng một từ có thể được gán các biểu diễn khác nhau tùy thuộc vào ngữ cảnh.


<!--
This motivates the development of *context-sensitive* word representations, where representations of words depend on their contexts.
Hence, a context-sensitive representation of token $x$ is a function $f(x, c(x))$ depending on both $x$ and its context $c(x)$.
Popular context-sensitive representations include TagLM (language-model-augmented sequence tagger) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe (Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017`, and ELMo (Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`.
-->

Điều này thúc đẩy sự phát triển của các biểu diễn từ *nhạy ngữ cảnh* (_context-sensitive_), trong đó biểu diễn của từ phụ thuộc vào ngữ cảnh của từ đó.
Do đó, biểu diễn nhạy ngữ cảnh của một token bất kỳ $x$ là hàm $f(x, c(x))$ phụ thuộc vào cả từ $x$ lẫn ngữ cảnh của từ là $c(x)$. 
Các biểu diễn nhạy ngữ cảnh phổ biến bao gồm TagLM (Bộ Tag chuỗi được tăng cường với mô hình ngôn ngữ (_language-model-augmented sequence tagger_)) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe (vector ngữ cảnh (_Context Vectors_)) :cite:`McCann.Bradbury.Xiong.ea.2017`, và ELMo (embedding từ các mô hình ngôn ngữ (_Embeddings from Language Models_)) :cite:`Peters.Neumann.Iyyer.ea.2018`.


<!--
For example, by taking the entire sequence as the input, ELMo is a function that assigns a representation to each word from the input sequence.
Specifically, ELMo combines all the intermediate layer representations from pretrained bidirectional LSTM as the output representation.
Then the ELMo representation will be added to a downstream task's existing supervised model
as additional features, such as by concatenating ELMo representation and the original representation (e.g., GloVe) of tokens in the existing model.
On one hand, all the weights in the pretrained bidirectional LSTM model are frozen after ELMo representations are added.
On the other hand, the existing supervised model is specifically customized for a given task.
Leveraging different best models for different tasks at that time, adding ELMo improved the state of the art across six natural language processing tasks:
sentiment analysis, natural language inference, semantic role labeling, coreference resolution, named entity recognition, and question answering.
-->

Ví dụ, ELMo là hàm gán một biểu diễn cho mỗi từ của chuỗi đầu vào bằng cách lấy toàn bộ chuỗi làm đầu vào cho hàm.
Cụ thể, ELMo kết hợp tất cả các biểu diễn tầng trung gian từ LSTM hai chiều đã được tiền huấn luyện làm biểu diễn đầu ra.
Sau đó, biểu diễn ELMo sẽ được đưa vào mô hình giám sát cho các tác vụ khác như một đặc trưng bổ sung, chẳng hạn bằng cách ghép nối biểu diễn ELMo và biểu diễn gốc (ví dụ GloVe) của token trong mô hình hiện tại.
Một mặt, tất cả các trọng số trong mô hình LSTM hai chiều được tiền huấn luyện đều bị đóng băng sau khi các biểu diễn ELMo được thêm vào.
Mặt khác, mô hình có giám sát được tùy biến cụ thể cho một tác vụ nhất định.
Thêm ELMo vào các mô hình tân tiến nhất cho các tác vụ khác nhau tại thời điểm ELMo được công bố giúp cải thiện chất lượng các mô hình này trên sáu tác vụ xử lý ngôn ngữ tự nhiên đó là:
phân tích cảm xúc (_sentiment analysis_), suy luận ngôn ngữ tự nhiên (_natural language inference_), dán nhãn vai trò ngữ nghĩa (_semantic role labeling_), phân giải đồng tham chiếu (_coreference resolution_) nhận dạng thực thể có tên (_named entity recognition_) và trả lời câu hỏi (_question answering_).


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## From Task-Specific to Task-Agnostic
-->

## Từ Đặc thù Tác vụ đến Không phân biệt Tác vụ


<!--
Although ELMo has significantly improved solutions to a diverse set of natural language processing tasks,
each solution still hinges on a *task-specific* architecture.
However, it is practically non-trivial to craft a specific architecture for every natural language processing task.
The GPT (Generative Pre-Training) model represents an effort in designing 
a general *task-agnostic* model for context-sensitive representations :cite:`Radford.Narasimhan.Salimans.ea.2018`.
Built on a Transformer decoder, GPT pretrains a language model that will be used to represent text sequences.
When applying GPT to a downstream task, the output of the language model will be fed into an added linear output layer
to predict the label of the task.
In sharp contrast to ELMo that freezes parameters of the pretrained model,
GPT fine-tunes *all* the parameters in the pretrained Transformer decoder during supervised learning of the downstream task.
GPT was evaluated on twelve tasks of natural language inference, question answering, sentence similarity, and classification,
and improved the state of the art in nine of them with minimal changes to the model architecture.
-->

Mặc dù ELMo đã cải thiện đáng kể giải pháp cho một loạt các tác vụ xử lý ngôn ngữ tự nhiên,
mỗi giải pháp vẫn dựa trên một kiến ​​trúc *đặc thù cho tác vụ* (_task-specific_).
Tuy nhiên, trong thực tế, xây dựng một kiến ​​trúc đặc thù cho mỗi tác vụ xử lý ngôn ngữ tự nhiên là điều không đơn giản.
Phương pháp GPT (Generative Pre-Training) thể hiện nỗ lực thiết kế một mô hình *không phân biệt tác vụ* (_task-agnostic_) chung cho các biểu diễn nhạy ngữ cảnh :cite:`Radford.Narasimhan.Salimans.ea.2018`.
Được xây dựng dựa trên bộ giải mã Transformer, GPT tiền huấn luyện mô hình ngôn ngữ được sử dụng để biểu diễn chuỗi văn bản.
Khi áp dụng GPT cho một tác vụ hạ nguồn, đầu ra của mô hình ngôn ngữ sẽ được truyền tới một tầng đầu ra tuyến tính được bổ sung
để dự đoán nhãn cho tác vụ đó.
Trái ngược hoàn toàn với cách ELMo đóng băng các tham số của mô hình đã được tiền huấn luyện,
GPT tinh chỉnh *tất cả* các tham số trong bộ giải mã Transformer đã được tiền huấn luyện trong suốt quá trình học có giám sát trên tác vụ hạ nguồn.
GPT được đánh giá trên mười hai tác vụ về suy luận ngôn ngữ tự nhiên, trả lời câu hỏi, độ tương tự của câu, và bài toán phân loại, và cải thiện kết quả tân tiến nhất của chín tác vụ với vài thay đổi tối thiểu đối tới kiến ​​trúc mô hình.


<!--
However, due to the autoregressive nature of language models, GPT only looks forward (left-to-right).
In contexts "i went to the bank to deposit cash" and "i went to the bank to sit down", as "bank" is sensitive to the context to its left,
GPT will return the same representation for "bank", though it has different meanings.
-->

Tuy nhiên, do tính chất tự hồi quy của các mô hình ngôn ngữ, GPT chỉ nhìn theo chiều xuôi (từ trái sang phải).
Trong các ngữ cảnh "i went to the bank to deposit cash" ("tôi đến ngân hàng để gửi tiền mặt") và "i went to the bank to sit down"("tôi ra bờ hồ để ngồi"), do từ "bank" nhạy với ngữ cảnh bên trái,
GPT sẽ trả về cùng một biểu diễn cho từ "bank", mặc dù nó có các ý nghĩa khác nhau.


<!--
## BERT: Combining the Best of Both Worlds
-->

## BERT: Kết hợp những Điều Tốt nhất của cả Hai Phương pháp


<!--
As we have seen, ELMo encodes context bidirectionally but uses task-specific architectures; while GPT is task-agnostic but encodes context left-to-right.
Combining the best of both worlds, BERT (Bidirectional Encoder Representations from Transformers)
encodes context bidirectionally and requires minimal architecture changes for a wide range of natural language processing tasks :cite:`Devlin.Chang.Lee.ea.2018`.
Using a pretrained Transformer encoder, BERT is able to represent any token based on its bidirectional context.
During supervised learning of downstream tasks, BERT is similar to GPT in two aspects.
First, BERT representations will be fed into an added output layer, with minimal changes to the model architecture depending on nature of tasks,
such as predicting for every token vs. predicting for the entire sequence.
Second, all the parameters of the pretrained Transformer encoder are fine-tuned, while the additional output layer will be trained from scratch.
:numref:`fig_elmo-gpt-bert` depicts the differences among ELMo, GPT, and BERT.
-->


Như ta đã thấy, ELMo mã hóa ngữ cảnh theo hai chiều nhưng sử dụng các kiến ​​trúc đặc thù cho tác vụ; trong khi đó GPT có kiến trúc không phân biệt tác vụ nhưng mã hóa ngữ cảnh từ trái sang phải.
Kết hợp những thứ tốt nhất của cả hai phương pháp trên, BERT (biểu diễn bộ mã hóa hai chiều từ Transformer)
mã hóa ngữ cảnh theo hai chiều và chỉ yêu cầu vài thay đổi kiến ​​trúc tối thiểu cho một loạt các tác vụ xử lý ngôn ngữ tự nhiên :cite:`Devlin.Chang.Lee.ea.2018`.
Sử dụng bộ mã hóa Transformer được tiền huấn luyện, BERT có thể biểu diễn bất kỳ token nào dựa trên ngữ cảnh hai chiều của nó.
Trong quá trình học có giám sát trên các tác vụ hạ nguồn, BERT tương tự như GPT ở hai khía cạnh.
Đầu tiên, các biểu diễn BERT sẽ được truyền vào một tầng đầu ra được bổ sung, với những thay đổi tối thiểu tới kiến ​​trúc mô hình tùy thuộc vào bản chất của tác vụ,
chẳng hạn như dự đoán cho mỗi token hay dự đoán cho toàn bộ chuỗi.
Thứ hai, tất cả các tham số của bộ mã hóa Transformer được đào tạo trước đều được tinh chỉnh, trong khi tầng đầu ra bổ sung sẽ được huấn luyện từ đầu.
:numref:`fig_elmo-gpt-bert` mô tả những điểm khác biệt giữa ELMo, GPT, và BERT.


<!--
![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
-->

![So sánh giữa ELMO, GPT, và BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
BERT further improved the state of the art on eleven natural language processing tasks
under broad categories of i) single text classification (e.g., sentiment analysis), ii) text pair classification (e.g., natural language inference),
iii) question answering, iv) text tagging (e.g., named entity recognition).
All proposed in 2018, from context-sensitive ELMo to task-agnostic GPT and BERT,
conceptually simple yet empirically powerful pretraining of deep representations for natural languages have revolutionized solutions to various natural language processing tasks.
-->

*dịch đoạn phía trên*


<!--
In the rest of this chapter, we will dive into the pretraining of BERT.
When natural language processing applications are explained in :numref:`chap_nlp_app`,
we will illustrate fine-tuning of BERT for downstream applications.
-->

*dịch đoạn phía trên*


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```


<!--
## Input Representation
-->

## *dịch đoạn phía trên*
:label:`subsec_bert_input_rep`


<!--
In natural language processing, some tasks (e.g., sentiment analysis) take single text as the input,
while in some other tasks (e.g., natural language inference), the input is a pair of text sequences.
The BERT input sequence unambiguously represents both single text and text pairs.
In the former, the BERT input sequence is the concatenation of the special classification token “&lt;cls&gt;”,
tokens of a text sequence, and the special separation token “&lt;sep&gt;”.
In the latter, the BERT input sequence is the concatenation of “&lt;cls&gt;”, tokens of the first text sequence,
“&lt;sep&gt;”, tokens of the second text sequence, and “&lt;sep&gt;”.
We will consistently distinguish the terminology "BERT input sequence" from other types of "sequences".
For instance, one *BERT input sequence* may include either one *text sequence* or two *text sequences*.
-->

*dịch đoạn phía trên*


<!--
To distinguish text pairs, the learned segment embeddings $\mathbf{e}_A$ and $\mathbf{e}_B$
are added to the token embeddings of the first sequence and the second sequence, respectively.
For single text inputs, only $\mathbf{e}_A$ is used.
-->

*dịch đoạn phía trên*


<!--
The following `get_tokens_and_segments` takes either one sentence or two sentences as the input, 
then returns tokens of the BERT input sequence and their corresponding segment IDs.
-->

*dịch đoạn phía trên*


```{.python .input}
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
BERT chooses the Transformer encoder as its bidirectional architecture.
Common in the Transformer encoder, positional embeddings are added at every position of the BERT input sequence.
However, different from the original Transformer encoder, BERT uses *learnable* positional embeddings.
To sum up, :numref:`fig_bert-input` shows that the embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.
-->

*dịch đoạn phía trên*


<!--
![The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.](../img/bert-input.svg)
-->

![*dịch mô tả phía trên*](../img/bert-input.svg)
:label:`fig_bert-input`


<!--
The following `BERTEncoder` class is similar to the `TransformerEncoder` class as implemented in :numref:`sec_transformer`.
Different from `TransformerEncoder`, `BERTEncoder` uses segment embeddings and learnable positional embeddings.
-->

*dịch đoạn phía trên*


```{.python .input  n=2}
#@save
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```


<!--
Suppose that the vocabulary size is 10,000.
To demonstrate forward inference of `BERTEncoder`,
let us create an instance of it and initialize its parameters.
-->

*dịch đoạn phía trên*


```{.python .input  n=3}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```


<!--
We define `tokens` to be 2 BERT input sequences of length 8, where each token is an index of the vocabulary.
The forward inference of `BERTEncoder` with the input `tokens` returns the encoded result 
where each token is represented by a vector whose length is predefined by the hyperparameter `num_hiddens`.
This hyperparameter is usually referred to as the *hidden size* (number of hidden units) of the Transformer encoder.
-->

*dịch đoạn phía trên*


```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Pretraining Tasks
-->

## *dịch đoạn phía trên*
:label:`subsec_bert_pretraining_tasks`


<!--
The forward inference of `BERTEncoder` gives the BERT representation of each token of the input text and the inserted special tokens “&lt;cls&gt;” and “&lt;seq&gt;”.
Next, we will use these representations to compute the loss function for pretraining BERT.
The pretraining is composed of the following two tasks: masked language modeling and next sentence prediction.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Masked Language Modeling
-->

### *dịch đoạn phía trên*
:label:`subsec_mlm`


<!--
As illustrated in :numref:`sec_language_model`, a language model predicts a token using the context on its left.
To encode context bidirectionally for representing each token, BERT randomly masks tokens and uses tokens from the bidirectional context to predict the masked tokens.
This task is referred to as a *masked language model*.
-->

*dịch đoạn phía trên*


<!--
In this pretraining task, 15% of tokens will be selected at random as the masked tokens for prediction.
To predict a masked token without cheating by using the label, one straightforward approach is to always replace it with a special “&lt;mask&gt;” token in the BERT input sequence.
However, the artificial special token “&lt;mask&gt;” will never appear in fine-tuning.
To avoid such a mismatch between pretraining and fine-tuning, if a token is masked for prediction 
(e.g., "great" is selected to be masked and predicted in "this movie is great"), in the input it will be replaced with:
-->

*dịch đoạn phía trên*


<!--
* a special “&lt;mask&gt;” token for 80% of the time (e.g., "this movie is great" becomes "this movie is &lt;mask&gt;");
* a random token for 10% of the time (e.g., "this movie is great" becomes "this movie is drink");
* the unchanged label token for 10% of the time (e.g., "this movie is great" becomes "this movie is great").
-->

*dịch đoạn phía trên*


<!--
Note that for 10% of 15% time a random token is inserted.
This occasional noise encourages BERT to be less biased towards the masked token (especially when the label token remains unchanged) in its bidirectional context encoding.
-->

*dịch đoạn phía trên*


<!--
We implement the following `MaskLM` class to predict masked tokens in the masked language model task of BERT pretraining.
The prediction uses a one-hidden-layer MLP (`self.mlp`).
In forward inference, it takes two inputs: the encoded result of `BERTEncoder` and the token positions for prediction.
The output is the prediction results at these positions.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
#@save
class MaskLM(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
To demonstrate the forward inference of `MaskLM`, we create its instance `mlm` and initialize it.
Recall that `encoded_X` from the forward inference of `BERTEncoder` represents 2 BERT input sequences.
We define `mlm_positions` as the 3 indices to predict in either BERT input sequence of `encoded_X`.
The forward inference of `mlm` returns prediction results `mlm_Y_hat` at all the masked positions `mlm_positions` of `encoded_X`.
For each prediction, the size of the result is equal to the vocabulary size.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```


<!--
With the ground truth labels `mlm_Y` of the predicted tokens `mlm_Y_hat` under masks,
we can calculate the cross entropy loss of the masked language model task in BERT pretraining.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```


<!--
### Next Sentence Prediction
-->

### *dịch đoạn phía trên*
:label:`subsec_nsp`


<!--
Although masked language modeling is able to encode bidirectional context for representing words, it does not explicitly model the logical relationship between text pairs.
To help understand the relationship between two text sequences, BERT considers a binary classification task, *next sentence prediction*, in its pretraining.
When generating sentence pairs for pretraining, for half of the time they are indeed consecutive sentences with the label "True";
while for the other half of the time the second sentence is randomly sampled from the corpus with the label "False".
-->

*dịch đoạn phía trên*


<!--
The following `NextSentencePred` class uses a one-hidden-layer MLP to predict whether the second sentence is the next sentence of the first in the BERT input sequence.
Due to self-attention in the Transformer encoder, the BERT representation of the special token “&lt;cls&gt;” encodes both the two sentences from the input.
Hence, the output layer (`self.output`) of the MLP classifier takes `X` as the input, where `X` is the output of the MLP hidden layer whose input is the encoded “&lt;cls&gt;” token.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
#@save
class NextSentencePred(nn.Block):
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```


<!--
We can see that the forward inference of an `NextSentencePred` instance
returns binary predictions for each BERT input sequence.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```


<!--
The cross-entropy loss of the 2 binary classifications can also be computed.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```


<!--
It is noteworthy that all the labels in both the aforementioned pretraining tasks can be trivially obtained from the pretraining corpus without manual labeling effort.
The original BERT has been pretrained on the concatenation of BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015` and English Wikipedia.
These two text corpora are huge: they have 800 million words and 2.5 billion words, respectively.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
## Putting All Things Together
-->

## *dịch đoạn phía trên*


<!--
When pretraining BERT, the final loss function is a linear combination of both the loss functions for masked language modeling and next sentence prediction.
Now we can define the `BERTModel` class by instantiating the three classes `BERTEncoder`, `MaskLM`, and `NextSentencePred`.
The forward inference returns the encoded BERT representations `encoded_X`, predictions of masked language modeling `mlm_Y_hat`, and next sentence predictions `nsp_Y_hat`.
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
#@save
class BERTModel(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```


## Tóm tắt

<!--
* Word embedding models such as word2vec and GloVe are context-independent. 
They assign the same pretrained vector to the same word regardless of the context of the word (if any). 
It is hard for them to handle well polysemy or complex semantics in natural languages.
* For context-sensitive word representations such as ELMo and GPT, representations of words depend on their contexts.
* ELMo encodes context bidirectionally but uses task-specific architectures 
(however, it is practically non-trivial to craft a specific architecture for every natural language processing task); 
while GPT is task-agnostic but encodes context left-to-right.
* BERT combines the best of both worlds: it encodes context bidirectionally and requires minimal architecture changes for a wide range of natural language processing tasks.
* The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.
* Pretraining BERT is composed of two tasks: masked language modeling and next sentence prediction. 
The former is able to encode bidirectional context for representing words, while the later explicitly models the logical relationship between text pairs.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Why does BERT succeed?
2. All other things being equal, will a masked language model require more or fewer pretraining steps to converge than a left-to-right language model? Why?
3. In the original implementation of BERT, the position-wise feed-forward network in `BERTEncoder` (via `d2l.EncoderBlock`) 
and the fully-connected layer in `MaskLM` both use the Gaussian error linear unit (GELU) :cite:`Hendrycks.Gimpel.2016` as the activation function.
Research into the difference between GELU and ReLU.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/388)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Văn Quang

<!-- Phần 2 -->
* Nguyễn Văn Quang

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

<!-- Phần 6 -->
* 

<!-- Phần 7 -->
* 
