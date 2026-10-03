# Suspicious Potion

> You shake the bottle, colored liquid sloshes around inside
> clinging to the insides of the glass. Every survival instinct
> tells you not to drink it except one: curiosity

This code is ill-advised for a few reasons.

First, it's generating a sentence embedding in JavaScript. Not through a native library like ONNX runtime. It's actually doing the math in JavaScript, which frankly you should not do.

It's not going to be fast and it might not even be correct.

I'm also loading the safetensors file in JavaScript with no dependencies, that seems silly and I probably got it wrong.

The model that we're using, is kind of a brightly colored frog too. It's part of the Potion family.

Potion is a family of static models created through Model2Vec. They use techniques such as Principal Component Analysis (PCA) and Zipf encoding to generate these sentence-transformer models that function without any forward inference.

This makes them fast and small, with a pretty modest drop in quality. They still score really well across the standard benchmarks for similarity and retreival.

The Minish Lab models range from 30MB (tiny) to just 8MB (atomic)! The model we're using here is even smaller. [potion-mxbai-micro](https://huggingface.co/blobbybob/potion-mxbai-micro) from certified maniac "blobbybob" weighs in at only 700KB.

[He's written about the process he used to get there](https://safereddit.com/r/LocalLLaMA/comments/1sapdue/700kb_embedding_model_that_actually_works_built_a/)

Because this model does no forward inference, it doesn't really use any matrix operations. There's a bit of addition, and then to normalize we have to calculate a cross product of two vectors and that's about it!

That means we can actually generate embeddings without native code or GPU powered kernels, or even a strong grasp on machine learning basics.

Get the data-set from here to follow along

```sh
curl -L -J --output-dir="./data" -O https://huggingface.co/datasets/rajistics/million-headlines/resolve/main/abcnews-date-text_train.csv
```