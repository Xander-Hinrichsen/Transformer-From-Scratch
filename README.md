# Transformer From Scratch
*This code is intended to be used for better understanding of the Transformer architecture, as it is not as optimized or time efficient as the PyTorch implementation of the Transformer.

Transformer implementation [Architectures/Transformer.py](Architectures/Transformer.py) from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), in 350 lines of PyTorch. \
Trained on only 87k Questions and Answers from the [Stanford Questions and Answers Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) *Due to GPU constraints



<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" style="display: block; margin: 0 auto; width: 420px; height: 500px;">

# Experiments
* Train everything completely from scratch (Scratch Model)
* Use and Freeze pretrianed BERT embeddings, but train everything else from scratch (BERT Embeddings)

# Results 

## Crystalline Knowledge Questions
Question: what is math ?\
Scratch Model Answer: $ connecting million\
Bert Embedding Answer: a system of knowledge, algebra, science and technology

Question: who was the first president of the united states ?\
Scratch Model Answer: president johnson\
Bert Embedding Answer: george washington

Question: what century was george washington from ?\
Scratch Model Answer: 19th\
Bert Embedding Answer: 18th

Question: who is george washington\
Scratch Model Answer: washington , later\
Bert Embedding Answer: one of the most of the president of wales is the united states

Question: what is the capital of france ?\
Scratch Model Answer: france\
Bert Embedding Answer: paris

Question: what continent is largest ?\
Scratch Model Answer: lake custom\
Bert Embedding Answer: antarctica

## Existential questions
Question: what do you think about darwinism ?\
Scratch Model Answer: is not resin (I thought this was funny)\
Bert Embedding Answer: he believed would be " tamed " by genomics policy which occurred, and to be defensive solutions. "

Question: is fast food good for the soul ?\
Scratch Model Answer: food and 9 per verify\
Bert Embedding Answer: dietary of flesh

Question: petrol or coal or oil\
Scratch Model Answer: less veterans\
Bert Embedding Answer: renewable
