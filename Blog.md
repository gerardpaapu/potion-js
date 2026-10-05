# Suspicious Potions and Brightly Coloured Frogs

As a programmer I'm easily distracted by novelty and I have a tendency to look at the equivalent of a brightly coloured frog and think about lunch. This is a story of something I found when I was meant to be looking for something else and the rabbit-hole I chased it down.

## What I was meant to be doing

Recently I was working on my note store `ruru` (you can skip some of this if you've read the full story). In particular I think there's opportunity to improve the semantic search functionality.

When I store a note in `ruru` I generate a list of questions that note answers. So if the article is something like:

```md
# Setting up your development environment

Start by running git clone ...
```

I'll generate (or write with my ape hands) questions like:
- How do I set up my development environment?
- How do I run tests?
- What are our dependencies?
- Can I run the project on Windows?

I associate the questions with the note in the database, and I generate an embedding for each question. An [embedding](https://en.wikipedia.org/wiki/Embedding_(machine_learning)) is a representation of some object as an array of numbers, usually that preserves some meaningful property.

### Embedding is making a thing that's like another thing in some way (in space)

In this case the property that I'm trying to preserve is "sentence similarity", so if two sentences are similar to each other their embeddings should be close to each in space.

I tend to imagine this literally, if you had a post-it for each sentence and you tried to lay them out on your desk in a way that more similar sentences were near each other and less similar sentences were further apart, you would be generating an embedding in 2d space (the surface of the desk). It's not going to work perfectly because the position on the desk just doesn't have as much information as each sentence does, but it does mean that someone else can tell which sentences you think are the most similar at a glance, even if they can't read.

So when I'm looking for a note in `ruru` I typically don't search by the text of the note itself. I typically search by asking a question.

```
$ ruru ask "How do I show that a button is disabled?"
```

`ruru` will encode the question into an embedding then it searches through my database for questions in my index that are "similar". For the most similar questions it lists the titles of the notes associated with them. So I get a list of notes that answer "questions like the one I asked" and I can pick one to read.

```
$ ruru ask "How do I show that a button is disabled?"
🟢 0.74: 4-button-jcdrd         | 4 - Button
🟢 0.72: 1-button-b9q6j         | 1 - Button: Features of a button
🟡 0.59: button-d78hl           | Button
🟡 0.53: disabled-read-sevv7    | Disabled and read-only

$ ruru cat disabled-read-sevv7
---
version: 2
updated_at: 2026-09-16T03:26:17Z
---

# Disabled and read-only

> **Introduction**: Disabled states are used to inform users that a particular interaction or action is presently unavailable. However, if an interactive element is disabled without proper context, blah blah blah a11y
```

There's [an sqlite3 extension ](https://github.com/asg017/sqlite-vec) that can do **very fast** search over a lot of vectors for similarity. This gives me semantic search with very little code. Modern coding agents tend to be pretty good at asking this kind of question so this helps them discover relevant information during tasks without overloading their context.

## Looking for a new model

In `ruru` I'm using `all-MiniLM-L6-v1` as my embedding model, it's a small model (87 MB on disk) that loads fast and can happily run in CPU. It generates an array of 384 32-bit floats for each "sentence", and it does an okay job.

If we're being honest with ourselves the reason I'm using it is because it's the textbook example of a sentence similarity model and I don't know much about what else is available. It turns out that if you look at [the leaderboards](https://huggingface.co/spaces/mteb/leaderboard) for this sort of thing there are lots of other (newer) models that are:

1. also open-source and freely available
2. about the same size on disk
3. fine-tuned for retrieval specifically
4. just better probably?

So my homework was: pick a few to compare, decide on some criteria, run some tests and find a model would replace `all-MiniLM-L6-v1` in `ruru` and in my heart.

Now I'm not saying this is a way to be but I've never been the best at doing my homework.

## Getting distracted by something shiny

I started out looking for better (usually a bit bigger) alternatives to `all-MiniLM-L6-v1` but in my search I came across [the Potion family](https://github.com/MinishLab/model2vec/tree/main) of static sentence-transformer models. What if instead of "bigger and better" we could have "smaller and worse"? I can tell you're intrigued.

Model2Vec is a technique that produces models that are **much** smaller and only score a **little** worse in accuracy.

>  Model2Vec reduces model size by a factor up to 50 and makes models up to 500 times faster, with a small drop in performance. 

These models are called "static" because they don't use a forward pass (dynamic computation) at all. This is how the docs describe their inference process.

> It does \[inference\] by computing one fixed vector per token, plus lightweight post-processing. Sentence embeddings are then produced by simply averaging token vectors.

If you've met me you can tell I've completely forgotten how I got here and that I was meant to be doing something for `ruru`. One of my fatal flaws is that I love solutions way more than I love problems.

Minish Labs' [smallest official model](https://huggingface.co/minishlab/potion-base-8M/tree/main) is only 30MB on disk. That's tiny.

Falling down the rabbit hole further I found [a true maniac (affectionate)](https://safereddit.com/r/LocalLLaMA/comments/1sapdue/700kb_embedding_model_that_actually_works_built_a/) has been training, distilling and quantizing in his lab inside an active volcano somewhere. He's been producing potion models that are even smaller than Minish Lab's official releases down to 700 kilobytes and **still doing pretty well on benchmarks**.

700K is so small! I downloaded it (didn't take long) and ran the sample code in python.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("blobbybob/potion-mxbai-micro")
embeddings = model.encode(["Hello world", "Static embeddings are fast"])
```

It really is fast, and it produced pretty good results to the naked eye. Internally `sentence_transformers` uses PyTorch to do all the matrix math and GPU magic. Model2Vec also has a runner that's based on numpy, because it doesn't need all the features of PyTorch since the architecture of the static models is simpler.

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("blobbybob/potion-mxbai-micro")
embeddings = model.encode(["Hello world", "Static embeddings are fast"])
```

## Either underestimating the problem or overestimating myself

This is when I had my "I should buy a boat" moment.

This model is tiny that I could easily load it into memory in JavaScript. The inference process is so **simple** that even I could probably implement it in JavaScript!

According to the liar who wrote their marketing, all I have to do:
1. compute one fixed vector per token
2. lightweight post-processing (whatever that means)
3. simply average token vectors

I can do **that**.

What's more is I'll do it without any third-party libraries or native extensions. Real slow serial single-core CPU JavaScript just like Hatsune Miku intended when [she invented it at Mozilla](https://knowyourmeme.com/memes/hatsune-miku-created-minecraft).

> [!NOTE]
>  If you're not familiar with this specific dramatic device, at this point I don't even know what some of these words mean let alone how to implement them or what's missing from this high-level description

## Token representation

Of the few things I know about these machines is:
1. they convert text into an array of integers (aka token ids)
2. those integers almost but don't quite map to one word each

In the model folder there's a `tokenizer.json` file. JSON is by defintion a human-readable format, so as a human I should be able to read it and intuit what to do next.

The `tokenizer.json` starts like this. I don't know what any of these words mean, so I'm just going to scroll down until something makes sense.

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": {
    "strategy": {
      "Fixed": 0
    },
    "direction": "Right",
    "pad_to_multiple_of": null,
    "pad_id": 0,
    "pad_type_id": 0,
    "pad_token": "[PAD]"
  },
```

In `.model.vocab` you start getting character sequences mapped to integers. At first I thought they were in order of length but as you go down the list it's obviously not true. 

I suspect they're actually in order of how often they show up in the training data, i.e. how commonly used they are in real human texts. 

```json
{
  "[PAD]": 0,
  "[UNK]": 1,
  "!": 2,
  "\"": 3,
  "#": 4,
  "$": 5,
  ...
  "unofficial": 10985,
  "##lies": 10986,
  "defunct": 10987,
  "eds": 10988,
  "moonlight": 10989,
  "drainage": 10990,
  "surname": 10991,
  ...
}
```

Some of the entries start with `##`, like `"##lies": 10986`. I'm pretty sure these are suffixes 🤔. This vocab does have the word `"week": 1736,` but it doesn't have "weeklies". So if we see "weeklies" I think we're meant to combine those two and emit `[1736, 10986]` for that word.

I put some words through the real implementation so I could compare the output to mine.

```python
model = StaticModel.from_pretrained("blobbybob/potion-mxbai-micro")
# "clockwork" is split to "clock" + "##work"
tokenized = model.tokenize(["just like clockwork"])
print(tokenized)
```
```sh
$ uv run potion-py
[[1077, 1069, 4122, 5201]]
```

**Some** compound words are split like that, but some like `"moonlight": 10989` should be read as a single token. So what we want is to match the whole word if it exists in the vocab and piece it together from a word and some suffixes if it doesn't.

My strategy is basically:

1. split the text on whitespace into "whole words".
1. look for each whole word in the vocab
2. ... or look for the longest prefix of that word that is in the vocab
3. from the end of that prefix look for the rest of the word
4. continue over all the words

```js
// start by considering the whole word
let start = 0;
let end = word.length;

// yield subwords until we cover the whole word
while (start < word.length) {
  // in 2026 I should learn .substr()
  let prefix = word.slice(start, end);
  let key = prefix;
  if (start !== 0) {
    // we're looking for a suffix
    key = `##${key}`
  }

  if (key in vocab) {
    yield vocab[key];
    // start looking for the next subword
    start += prefix.length;
    end = word.length;
    continue;
  }
  
  // let's try a shorter prefix
  end--;
}
```
... and it basically just works!

```js
> tokenize("just like clockwork", tokenizer)
[ 1077, 1069, 4122, 5201 ]
```

I'm nailing it. This is easy! What's next? I need to load the model _somehow_ and encode those token ids to an embedding.

## Encoding tokens to an embedding

We can look at our python reference again to know what we want to end up with.

```python
model = StaticModel.from_pretrained("blobbybob/potion-mxbai-micro")
embeddings = model.encode(["just like clockwork"])
print(embeddings[0])
```
```sh
$ uv run potion-py 
[
  0.07166269421577454,
  0.13574431836605072,
  -0.09292009472846985,
  0.02098734676837921,
  -0.05866039916872978,
  0.043105270713567734,
```

So the result we're looking for is an array of 256 floats, and they should be pretty much the same as these ones. (You usually can't expect any two floats to be exactly equal).

I've got this `model.safetensors` file, and genuinely no idea what a "safetensors" file looks like on the inside. Usually they're just a [black box](https://en.wikipedia.org/wiki/Black-box_testing). I pass the filename to a library written by someone who knew what they were doing and everything Just Works.

Since it's a way to package huge blobs of binary data, I assume it looks a little like [cdb](https://cr.yp.to/cdb/cdb.txt) on the inside? But the blobs also have different shapes and purposes, so maybe it's like [the inside of a tar file](https://en.wikipedia.org/wiki/Tar_(computing)#UStar_format)?

There's two ways to find out and **today we're doing the stupid one!**.

Let's just have a peek inside.

## Looking inside a Safetensors file

If you're not familiar with `xxd` it's a "hexdump" tool that's probably installed on your machine already. It shows the bytes of a file in hexadecimal, and also the ascii bytes as the characters.

So this is the first 300 bytes of `model.safetensors`.

```
$ head -c 300 < ~/.local/share/models/potion-mxbai-micro/model.safetensors | xxd
00000000: e000 0000 0000 0000 7b22 7765 6967 6874  ........{"weight
00000010: 7322 3a7b 2264 7479 7065 223a 2246 3332  s":{"dtype":"F32
00000020: 222c 2273 6861 7065 223a 5b32 3935 3235  ","shape":[29525
00000030: 5d2c 2264 6174 615f 6f66 6673 6574 7322  ],"data_offsets"
00000040: 3a5b 302c 3131 3831 3030 5d7d 2c22 6d61  :[0,118100]},"ma
00000050: 7070 696e 6722 3a7b 2264 7479 7065 223a  pping":{"dtype":
00000060: 2249 3332 222c 2273 6861 7065 223a 5b32  "I32","shape":[2
00000070: 3935 3235 5d2c 2264 6174 615f 6f66 6673  9525],"data_offs
00000080: 6574 7322 3a5b 3131 3831 3030 2c32 3336  ets":[118100,236
00000090: 3230 305d 7d2c 2265 6d62 6564 6469 6e67  200]},"embedding
000000a0: 7322 3a7b 2264 7479 7065 223a 2249 3822  s":{"dtype":"I8"
000000b0: 2c22 7368 6170 6522 3a5b 3230 3030 2c32  ,"shape":[2000,2
000000c0: 3536 5d2c 2264 6174 615f 6f66 6673 6574  56],"data_offset
000000d0: 7322 3a5b 3233 3632 3030 2c37 3438 3230  s":[236200,74820
000000e0: 305d 7d7d 2020 2020 5418 003f 7424 003f  0]}}    T..?t$.?
000000f0: 1201 963e 07f4 3e3e 40b6 973e 9bf2 c83e  ...>..>>@..>...>
00000100: 151e b43e 065c 873e 906f 833e 3643 1e3e  ...>.\.>.o.>6C.>
00000110: b153 403e 999f 753e 538a ac3e 390b 523e  .S@>..u>S..>9.R>
00000120: 8622 573e a822 773e 8d29 303e            ."W>."w>.)0>
```

I was pretty surprised (but happy) to see my old friend JSON in this big bag of bytes.

When decoding binary formats, you usually want to know how many bytes to read up front. So the sections are either fixed-length or they have a fixed-length prefix that tells you how many bytes to read for the section.

In our `.safetensors` file the first two bytes are `e0` and then the next 6 bytes are zeroes.

We've got 8 bytes here and we know we're looking for at least one unsigned integer because byte-lengths are positive whole numbers.

It could be 4 16-bit integers, 2 32-bit integers or even 1 64 bit integer. In theory it could even be am 8 bit integer, but that seems unlikely because it would limit the length to a maximum of 255 bytes.

For loading **this specific** model, it doesn't really matter which is true. Here's what it looks like if we try all the options on a buffer with those 8-bytes.

```js
> let raw = Buffer.from([0xe0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
undefined
> new Uint16Array(raw.buffer, 0, raw.length / Uint16Array.BYTES_PER_ELEMENT)
Uint16Array(4) [ 224, 0, 0, 0 ]
> new Uint32Array(raw.buffer, 0, raw.length / Uint32Array.BYTES_PER_ELEMENT)
Uint32Array(2) [ 224, 0 ]
> new BigUint64Array(raw.buffer, 0, raw.length / BigUint64Array.BYTES_PER_ELEMENT)
BigUint64Array(1) [ 224n ]
```

No matter which way we cut it, there's a `224` and "some zeroes".

The JSON we saw in xxd starts at byte 9 with `'{"weight'` and ends at byte 228 with `'}'0]}}'` then the next 8 bytes are `0x20` aka space `" "`. 

If I read 224 bytes starting at 9 we end up taking _half_ of those spaces.

"Half" is a little spooky. "All" or "none" would both feel like we're doing the right thing but `JSON.parse()` will just ignore trailing spaces so everything's fine!

Ignore that spooky feeling and charge ahead. That weirdly specific amount of padding is probably there for no reason!

## Reading is what? Fundamental

I don't use them often but `node:fs` does come with all the functions you need to read a specific number of bytes from specific byte locations in a file.

`await fs.read(fileHandle, buffer, offset, length, position)` reads `length` bytes from `position` in the file and writes them to the location starting at `offset` in `buffer`.

```js
await using safetensors = await FS.open(`${MODELS_DIR}/potion-mxbai-micro/model.safetensors`);

const prefix = new Uint32Array(2); // 2 number in 8 bytes
const PREFIX_BYTE_LENGTH = 8; // how many bytes is it
const PREFIX_BYTE_OFFSET = 0; // where is it in the file
await safetensors.read(prefix, 0, PREFIX_BYTE_LENGTH, PREFIX_BYTE_OFFSET);
const [headerByteLength, idk] = prefix; // [224, 0]
const HEADER_BYTE_OFFSET = 8; // starts right after the prefix
const headerRawBytes = new Uint8ClampedArray(headerBytesLength);

await safetensors.read(headerRawBytes, 0, headerByteLength, HEADER_BYTE_OFFSET);

// JSON is utf-8 by international law
const headerJson = headerRawBytes.toString('utf-8');
const header = JSON.parse(headerJson);

console.log(header);
```
 if we did everything right 🤞 it looks like this
```js
{
  weights: { dtype: 'F32', shape: [ 29525 ], data_offsets: [ 0, 118100 ] },
  mapping: { dtype: 'I32', shape: [ 29525 ], data_offsets: [ 118100, 236200 ] },
  embeddings: {
    dtype: 'I8',
    shape: [ 2000, 256 ],
    data_offsets: [ 236200, 748200 ]
  }
}
```
The straightforward reading of this is that it's describing sections in the file. We've got their names, shapes and where in the file they are.

Data offsets seems straightforward, they look like they're `[start, end]` as byte offsets. The weights start at `0` so we must be reading from after the header, so that's really at 8 + 224, and the next 118,100 bytes are "weights". 

`dtype` seems straightforwardly to map to C number types or JavaScript typed arrays. `F32` is 32 bit floats, `I32` is 32 bit integers, `I8` is 8 bit integers. I'm assuming signed for both the integer types. "d" probably stands for "duh".

Shape must be the dimensions of the tensor, so weights and mappings are just vectors (or normal arrays), and the embeddings are a 200x256 2D matrix.

For now let's just take those values and hardcode them into our loader.

```js
const dataStart = HEADER_BYTE_OFFSET + headerByteLength;
// starting from the end of the header, we use the data_offset values
const weightsOffset    = dataStart + 0      // data_offsets: [0, 118100]
const mappingsOffset   = dataStart + 118100 // data_offsets: [118100, 236200]
const embeddingsOffset = dataStart + 236200 // data_offsets: [236200, 748200]
const weights    = new Float32Array(29525);
const mappings   = new Int32Array(29525);
const embeddings = new Int8Array(512000);

await safetensors.read(weights,    0, 118100, weightsOffset);
await safetensors.read(mappings,   0, 118100, mappingsOffset);
await safetensors.read(embeddings, 0, 512000, embeddingsOffset);
```

## Staring into the void

So I think I've got the blobs, but no idea what to do with them. 

- "weights" usually means I'm multiplying something by something else.
- "mappings" sounds like a table mapping some domain to some other domain in some straightforward way.
- "embeddings" is what we want to end up with what does it mean to just have some lying around?

I spent a long time stuck here, you can simulate that experience by just squinting at this sentence for 45 minutes before continuing. It probably helps my engagement stats too.

What eventually got me unstuck was noticing that both mappings and weights have length 29,525.

It's not a power of two, or any other magic number I'm familiar with. So I'm looking everywhere for another reason why it would be that exact number. Finally I logged out the number of keys in the vocab. **There are 29,525 keys in the vocab.**

Of course the ids in the vocab aren't just random ints! They must be indexes! Every number between 0 and 29,524 points to a specific token.

"mappings" and "weights" are that exact size so I'm probably meant to lookup each token id in them like a table!

"Weight" is usually a factor you multiply something by and the only other number I have is the mapping. So maybe we're meant to multiply the mapping by the weight?

```js
for (let token_id of tokens) {
  const weight = weights[token_id]; // an f32
  const mapping = mappings[token_id]; // an int32

  something = weight * mapping; // maybe something like this ??
}
```

## What are the embeddings for?

This felt weird and it wasn't even using the embeddings. I guess it wouldn't make a lot of sense to have that value spread across to sources if you're just going to multiply them since the result is going to be an 32-bit float anyway. 

I spent a while logging values of `mapping` and `weight` out to see what was going on. I realised (after a long time) that the values of `mapping` are all between 0 and 1,999.

"embeddings" is 2,000 rows of 256 bytes. I think I'm meant to look these up in that? and I guess I multiply that whole row by `weight`?

This is basically what I'm picturing now.

```js
const EMBEDDING_SIZE = 256; // the length of each row

for (let token_id of tokens) {
  const weight = weights[token_id]; // an f32
  const row = mappings[token_id]; // an int32
  for (let i = 0; i < EMBEDDING_SIZE; i++) {
    ??? = embeddings[row * EMBEDDING_SIZE + i] * weight;
  }
}
```

So in theory, we're using everything that's there to use but we're not getting anything out yet. We know what we want out (because of the python version) is Float32Array (length = 256).

Referring back to the list that sent us down the rabbit hole, we're meant to take the average of that row.

1. ~~compute one fixed vector per token~~ (nailed it!)
2. lightweight post-processing (still no idea, let's ignore it)
3. simply average token vectors (I know what some of these words mean)

```js
const EMBEDDING_SIZE = 256; // the length of each row
// I just need this to look different to the word "embeddings"
const resultEmbedding = new Float32Array(256);

for (let token_id of tokens) {
  const weight = weights[token_id]; // an f32
  const row = mappings[token_id]; // an int32
  for (let i = 0; i < EMBEDDING_SIZE; i++) {
    // we end up with the sum of the i-th column from each token's embedding
    resultEmbedding[i] += embeddings[row * EMBEDDING_SIZE + i] * weight;
  }
}

for (let i = 0; i < EMBEDDING_SIZE; i++) {
  // we divide to get the mean
  resultEmbedding[i] /= tokens.length;
}
```

Okay, how's the list?

1. ~~compute one fixed vector per token~~
2. ~~lightweight post-processing~~ (it's probably fine)
3. ~~simply average token vectors~~

## Well what does it mean to be normal anyway?

At this point, I really thought I was doing everything right. My target was an array of 256 numbers and I was getting an array of 256 numbers. So I had a **moral** victory but my numbers were different to the `sentence-transformers` library's numbers by a lot.

I went to bed at this point. It was pretty late. 

In the morning I was having a look at [wikipedia](https://en.wikipedia.org/wiki/Embedding_(machine_learning)) and I found this:

> To measure the distance between two embeddings, a similarity measure can be used to find the overall similarity of the concepts represented by the embeddings. If the vectors are normalized to have a magnitude of 1, then the similarity measures are proportional to 

I guess in the ML space this is one of the things that's so obvious that people never say it explicitly, but it's not obvious to people that just don't know it (like me).

Embeddings are usually normalised to the same magnitude before being compared. This means that instead of calculating something like cosine similarity (which is expensive) you can use a distance calculation (which is less expensive). For vectors of the same magnitude `1 / distance` isn't the same as cosine similary, but it will have the same ranking. Which is all we care about for a question like "which embeddings are the most similar".

In fact even `1 / distance²` will have the same ranking and that's even cheaper to calculate.

```js
function normaliseL2(embedding: Float32Array) {
  // first we calculate the current magnitude/length
  let magnitude2 = 0;
  for (let i = 0; i < EMBEDDING_SIZE; i++) {
    magnitude2 += embedding[i] ** 2;
  }

  const magnitude = Math.sqrt(magnitude2);
  for (let i = 0; i < EMBEDDING_SIZE; i++) {
    embedding[i] /= magnitude;
  }

  // now the embedding is (roughly) unit length
}
```

I realise this is not going to be as exciting for you as it was for me, but my JavaScript program was logging the same(-ish) numbers as my python program.

```js
$ ./potion-js | jq
[
    0.07166268676519394,
    0.13574430346488953,
    -0.09292009472846985,
    0.02098734676837921,
    -0.058660395443439484, 
    0.043105266988277435,
```

Job done! Right! We did it! Right?

## If a tree falls in the forest but nobody was impressed, did it make a sound?

Not yet. First I need to write a demo. It's not interesting to just show someone that you can turn a text into some random numbers, we can all do that! I want to show people a working demo of those random numbers **doing something.**

The normal demo for this sort of thing is semantic search (that rings a bell, why were we looking at models again?). For semantic search you normally do something like this.

1. encode embeddings for all the objects in your search domain (haystack)
2. Put those in some kind of index
3. encode an embedding for the object you're searching for (needle)
4. do an Approximate Nearest Neighbours search to find the nearest objects in the haystack to your needle.

Steps 2 and 4 are the problem for this exercise.

Usually I would pull something off the shelf, like [FAISS](https://en.wikipedia.org/wiki/FAISS) or [sqlite-vec](https://github.com/asg017/sqlite-vec). Someone who knows what they're doing has already written a native library that makes it super fast to search for a similar embedding. They rely on clever math which I don't understand and GPU or SIMD tricks that JavaScript just doesn't have access to.

We're not pulling things off the shelf today or at least we're not pulling C++ libraries off the shelf to do all of our number crunching. It would be a betrayal of everything we've done to get this far.

If I had a small enough data set I could just put them in an array and calculate their distance from the needle 1-by-1 but nobody wants to see a demo of "Semantic search over a dataset of 25 headlines" it doesn't have the right ring to it. I want to demo "Semantic search over a **million headlines**." It's what the guy Justin Timberlake was playing in that movie would tell me to do.

So I downloaded [a data set of 1 million headlines](https://huggingface.co/datasets/rajistics/million-headlines/blob/main/abcnews-date-text_train.csv) and stared at it. Hoping that I could figure something out.

## I want the same reward... but I want to do less work

I needed some way to arrange these in advance so that at **runtime** I don't have to do 1,000,000 distance calculations or I need some way to make the distance calculations **much** faster.

My first thought was I could build an index by [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) the full set ahead of time.

> [!NOTE]
>
> If you've known me for a while, I have a pretty small bag of tricks.I tend to pull them out and hope they apply to the problem at hand. K-means clustering is one I [learned a while ago](https://www.oreilly.com/library/view/programming-collective-intelligence/9780596529321/) and used when we were trying to figure out which "xero blue" is the **real** xero blue.
>
> So now when I get stuck in a meeting I might just say "have we we tried k-means clustering" even if it's completely irrelevant it has smart sounding words

This would give me some much smaller number of points in the space to test against, aka. "centroids" and a list of each object in the index near that centroid. So if I ran clustering with `k = 12`, I could do the expensive distance calculation over only 12 objects, take the nearest two and search their lists. 

But at `k = 12`, each centroid would still have like 80,000 children

I think the break even is something like `k = 1_400` where each centroid would have (on average) 700 children. So instead of doing 1,000,000 expensive comparisons I would be doing (on average) 2,800 comparisons.

Doing almost 3k comparisons at runtime doesn't sound fast and building the index would require **millions** of slow comparisons.

I gave up on this strategy.

## Quantity vs Quality

The other route is finding a cheaper (but less accurate) way to calculate to rank the candidates, find the top 100 (for example) candidates by that metric. Then re-rank those by the accurate expensive metric to choose the top 10.

This relies on the cheap metric having some fairly strong correlation with the expensive metric, and making first cut bigger than the second so that the odds of any member of the "true" top 10 being outside of the approximate top 100 is low.

Hunting around for a cheap version of vector distance I came across a vaguely familiar word "quantization". When you're doing all this vector math business you can get similar but less accurate results for cheaper by scaling down from 32 bits per dimension or 8bits, 4bits or ... even 1 bit per dimension.

8 bit numbers or 4 bit numbers aren't really any cheaper or faster in a language like JavaScript. JavaScript uses 64 bit floats for every number so quantizing to 4-bits doesn't help.

Things only get interesting when we talk about quantizing to 1-bit per dimension, because we can start replacing normal math operations with bitwise operations.

Starting with our full size embedding, it's 256 32-bit floats. To turn each float into a bit we ask if the float is greater than 0. If it is that dimension is a 1, otherwise it's a 0.

Now our _quantized_ embedding is only 256 bits. We always want to represent bitfields in JavaScript as unsigned 32 bit integers, since that's what all the bitwise operators operate on. `256 = 8 x 32` so we can represent as an array of 8 32-bit signed integers.

```js
function quantizeTo1Bit(embedding) {
  const output = new Uint32Array(8);
  for (let i = 0; i < 256; i++) {
    const bitIndex = i % 32; 
    const intIndex = i >>> 3; // Math.floor(i / 8)
    const bit = embedding[i] > 0 ? 0 : 1;

    output[intIndex] |= (bit << bitIndex);
  }
  return output;
}
```

For two 32-bit integers `a` and `b`, if we XOR them `a ^ b`, and then count the 1s in the binary representation (aka. popcount), we get the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance). What's even better is Hamming distances are additive. So we can just pairwise calculate the Hamming distance between parallel sections of the 1-bit vectors and add them up.

```javascript
export function hammingDistance256(a, b) {
  return (
    popcount32(a[0] ^ b[0]) +
    popcount32(a[1] ^ b[1]) +
    popcount32(a[2] ^ b[2]) +
    popcount32(a[3] ^ b[3]) +
    popcount32(a[4] ^ b[4]) +
    popcount32(a[5] ^ b[5]) +
    popcount32(a[6] ^ b[6]) +
    popcount32(a[7] ^ b[7])
  );
}
```

In serious languages, you usually have access to popcount as [a single CPU instruction](https://en.wikipedia.org/wiki/SSE4#POPCNT_and_LZCNT). **Not Javascript though!**.

I found a ["fast" version for JavaScript](https://gist.github.com/leodutra/63ca94fe86dcffee1bab#file-bitwise-hacks-js-L227-L236). then I spent ages trying to get nerds on discord and Gemini to explain it to me until it made sense.

```javascript
function popCount(n) {
  n = n - ((n >>> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
  return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
```

[It's based on this C implemention for int 32](https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel)

<details>
<summary>The explanation is kind of a psychic attack so read at your own risk</summary>

First, let's look at the magic numbers in binary.

```
0x55555555: 01010101 01010101 01010101 01010101
0x33333333: 00110011 00110011 00110011 00110011
0x0F0F0F0F: 00001111 00001111 00001111 00001111
0x01010101: 00000001 00000001 00000001 00000001
```

The first 3 are used with `&` to mask out certain sets of binary digits, i.e. set them to zero. So don't think of them as numbers, think of them as a selection of bits. The last one is a treat for later.

```
Selects every other bit
0x55555555: █ █ █ █  █ █ █ █  █ █ █ █  █ █ █ █ 

Selects every other pair
0x33333333: ██  ██   ██  ██   ██  ██   ██  ██  

Selects every other 4-block 
0x0F0F0F0F: ████     ████     ████     ████    
```

Let's give these names. It's still confusing as heck but maybe a little easier to look at.

```js
const everyOtherBit  = (x) => x & 0x55555555;
const everyOtherPair = (x) => x & 0x33333333;
const everyOtherQuad = (x) => x & 0x0F0F0F0F;
const combineSumsInLeftmostByte = (x) => x * 0x01010101;
const shiftResultToRightmostByte = (x) => x >> 24;

function popCount(n) {
  // count the 1s in every 2-block
  let a = n - everyOtherBit(n >> 1);
  // sum each pair of 2-block
  let b = everyOtherPair(a) + everyOtherPair(a >>> 2);
  // sum each pair of 4-blocks
  let c = everyOtherQuad(b) + everyOtherQuad(b >>> 4);
  // sum all 4 bytes together. the result is in the left-most byte
  let d = combineSumsInLeftmostByte(c);
  // shift the final sum to the right-most byte
  let final = shiftResultToRightmostByte(d);
  return final;
}
```

In step 1. we're counting each pair in place. That is we're replacing the original value with the number of 1s. There's only 4 values that can be represented in a pair:

| value (bin) | value (dec) | count (dec) | count (bin) |
| -- | -- | -- | -- |
| `00` | `0` | `0` | `00` |
| `01` | `1` | `1` | `01` |
| `10` | `2` | `1` | `01` |
| `11` | `3` | `2` | `10` |

We'll use `23` as an example, in bits it looks like:

```
00000000 00000000 00000000 00010111
```

It has four `1`s (count them if you don't believe me) so we're looking for a final answer of `4`.

Since `23` is small we can go through the process with a focus on just the last byte `00010111`.

We'll follow the first line with our byte split into blocks of 2-bits. Each is translated through this process from the original bit pattern to a count of the `1`s.

Each block has two bits so the result can be zero `1`s = `00`, one `1` = `01` or two `1`s = `10`.

| value | pairs |
| -- | -- |
| n = 23 | `00 01 01 11` |
| 23 >> 1 | `00 00 10 11` |
| everyOtherBit(23 >> 1) | `00 00 00 01` |
| 23 - everyOtherBit(23 >> 1) | `00 01 01 10` |
| counts | `[0, 1, 1, 2]` |

We can eventually convince ourselves that the original eye-watering code is counting the number of `1`s in each pair and storing that count in the pair itself. (Take your time).

This means `a = 00010110` this is decimal 22 but the number value of the whole sequence doesn't matter right now. Instead think of it as being a series of 2-bit pairs representing these sums `[..., 0, 1, 1, 2]`.

The next two steps are combining those sums into larger blocks. 

Every block of 4 contains 2 pairs and we'll add those pairs up. `[..., 0, 1, 1, 2]` becomes `[..., 0 + 1, 1 + 2]`.

| value | 4-blocks |
| -- | -- |
| `everyOtherPair(a)` | `0001 0010` |
| `everyOtherPair(a >> 4)` | `0000 0001` |
| `everyOtherPair(a) + everyOtherPair(a >> 4)` | `0001 0011` |
| counts | `[1, 3]` | 

Then we want to turn `[..., 1, 3]` into `[...,  1 + 3]`.

| value | 8 bits |
| -- | -- |
| `everyOtherQuad(b)` | `00000011` |
| `everyOtherQuad(b >> 2)` | `00000001` |
| `everyOtherQuad(b) + everyOtherQuad(b >> 2)` | `00000100` |
| counts | `[..., 4]` | 

Now each byte contains their own population count. A byte is 8 bits long so every byte should contain one of these totals/bit-patterns.

```
0 | 00000000   3 | 00000011   6 | 00000110
1 | 00000001   4 | 00000100   7 | 00000111
2 | 00000010   5 | 00000101   8 | 00001000
```

The **trust me bro** here, is that we've been looking at the right-most byte of the 32-bit integer. So you just have to imagine that the same process has been happening to the other 3 bytes and the whole sequence looks like this.

```
00000000 00000000 0000000 00000100
```

That means our `[..., 4]` expands out to `[0, 0, 0, 4]`. For the next step we need to imagine that there are values in the other buckets. So let's consider a different scenario.

If we started with `00001111 00001111 00001111 00001111` we would have `[4, 4, 4, 4]` or in bits it would look like `00000100 00000100 00000100 00000100`.

We need to combine each bucket of 4 into one bucket of 16 `00010000`. That's where our last magic number comes in. If we multiply by `0x01010101` 

```
  00000100 00000100 00000100 00000100 aka. [4, 4, 4, 4]
x 00000001 00000001 00000001 00000001 aka. (0x01010101)
-------------------------------------
  00010000 00001100 00001000 00000100 
```

We get a number that has our final count in the left-most bucket. We no longer care what's in any other byte. We just want that final total.

To get it we shift every bit 24 spaces to the right `(>>> 24)`.

```
00000000 00000000 0000000 0000000 00010000 = 16
```
</details>

---

It definitely cost me a few sanity points but believe me when I tell you **it's really fast**.

## Putting it all together

So if we're starting with a 1 million item haystack, we can follow this process.

1. encode the haystack to 1KB (256 x 32 bits) embeddings
2. quantize the embeddings to 32 bytes (256 x 1 bit)
3. encode the needle to a 1KB embedding
4. quantize that embedding to 32 bytes
5. look through the whole haystack finding the closest 100 by Hamming Distance with the 32 byte embedding (cheap)
6. re-rank those 100 best candidates to find the top 10 by euclidian distance (expensive, but we only have to do it 100 times)

## Building our indexes

The important part here is that steps 1 and 2 can be done ahead-of-time. So we encode the 1 million embeddings and the quantizations and write them to separate files. Now we can re-use them for any number of searches over the same set.

The other thing worth mentioning for steps 1 and 2 is that in the final code, I changed all my typed array operations to take an offset.

```js
export function distanceSquared(needle, haystack, offset = 0) {
  const j = offset * 256; // each embedding is 256 entries
  let d2 = 0;
  for (let i = 0; i < 256; i++) {
    // haystack contains **all** the candidates so we read
    // 256 bytes starting from offset * 256
    const diff = needle[i] - haystack[j + i];

    d2 += diff ** 2;
  }

  return d2;
}
```

This means that instead of a million embeddings living in a million separate `Float32Array`s they can all live in huge array. Similarly for the quantized embeddings they can all live in one too. (In theory I could have put them both in the same one). This is meant to make the operations more cache friendly because I'm looping over a big contiguous regions of memory rather than pulling a new reference from the heap for each iteration.

Honestly, I don't know how well it worked but it feels good to make an effort.

Once I'd built both of those, I just wrote them to disk as two big unlabelled blobs. Probably not ideal.

## Searching the haystack

So now we do the two-pass search. Finding the top-100 with our fast-but-fuzzy metric (256-bit embeddings and hamming distance) and then re-ranking for the top-10 with our more accurate metric (1KB embeddings and euclidian distance).

Sorting n-items is famously `O(n log n)` in time so we don't want to do that if n = 1 million but finding the top-k doesn't actually require a full sort. In our case we keep a fixed length (sorted) array of the top-k items in a pass. Something like this.

```js
let top10 = [...TEN_TRASH_RESULTS];
for (const candidate of oneMillionCandidates) {
  const score = expensiveCalculation(candidate);
  const result = { score, candidate };
  if (score > top10[0].score) {
    top10[0] = result;
    // O(10 log 10) is just O(k)
    top10.sort(byScore);
  }
}
```

So we're not sorting the full million items. `k` is much smaller than `n` so we can consider the work to keep our final array in order as a constant time cost per candidate. This makes "taking the top-k from a n items" take linear time with respect to `n` aka it's `O(n)`.

Now of course we're actually doing this in two passes. The first is to reduce our search space to one that is:

1. much smaller than a million
2. very likely contains the real best 10

This means we need to remember the index we found our candidates at, so we can look at only those ones.

There's three parallel collections here.
1. slowEmbeddings - an array with 256 million 32 bit floats, these are our full embeddings
2. fastEmbeddings - an array with 4 million 32 bit integers, these are our quantified-to-1-bit aka. Q1 embeddings
3. source - an array of 1 million strings, these are the headlines we're searching

They're all in the same order, such that for any `idx in [0, 1_000_000)`

1. `slowEmbeddings[idx]` is the embedding for `source[idx]`
2. `fastEmbeddings[idx]` is the quantifiaction of `slowEmbeddings[idx]`

We load them each from a file and then start the search.

```js
const needle = `Ice cream is poisoned`;
let needleEmbedding = encode(needle);
let needleQ1Embedding = quantizeTo1Bit(needleEmbedding);

let top100 = [...ONE_HUNDRED_TRASH_RESULTS];

for (const [idx, candidate] of fastEmbeddings.entries()) {
  const score = fastScore(needleQ1Embedding, candidate);
  const result = { score, idx };
  
  // if we're better than the worst member of the top 100
  // we replace them
  if (score > top100[0].score) {
    top100[0] = result;
    // O(100 log 100) is a constant so this is O(k)
    // we keep the best at top100[99] and the worst at top100[0]
    top100.sort(byScoreAscending);
  }
}

// re-rank the top 100 taking the top 10
const top10 = [...TEN_TRASH_RESULTS];
for (const { idx } of top100) {
  const candidate = slowEmbeddings[idx];
  const score = slowScore(needleEmbedding, candidate);
  const result = { score, idx };

  if (score > top10[0].score) {
    top10[0] = result;
    top10.sort(byScoreAscending);
  }
}

// finally we look the top 10 up in the source
// and reverse them to get #1 ... #10
const final = top10.map(({ idx }) => source[idx]).toReversed();

for (const [i, headline] of final.entries()) {
  console.log(`${i.toString().padStart(2)}. ${headline}`);
}
```

On my machine this runs in about 10 seconds. I build both indexes in memory then write them all at once so it also sits on over a Gig of memory.

```
$ time ./build-indexes.js
Hit limit at 1000000
creating Embeddings for 1,000,000 headlines: 8.443s
Quantizing 1,000,000 embeddings: 1.357s
Writing index files: 703.523ms
./build-indexes.js  7.88s user 2.84s system 100% cpu 10.710 total
```

## Let's see the demo!

Searching on my machine takes 1.3s with these indexes

```
$ time ./search.js <<< 'Ice cream is poisoned'
 1. (0.86) Lawyers question source of ice cream poisoning
 2. (0.86) Man seeks compo over alleged ice cream poisoning
 3. (0.85) Toxic 2
 4. (0.85) Toxic food
 5. (0.84) Poison use drops
 6. (0.82) Poison warning
 7. (0.80) Toxic plant kills cattle
 8. (0.77) Toxic waste site row heats up
 9. (0.74) Toxic algae poisoning tasmania could harm businesses scientist
10. (0.65) Poisoned dogs
./search.js <<< 'Ice cream is poisoned'  0.34s user 1.04s system 105% cpu 1.314 total
```

(Sorry `#cop-frontend` I couldn't bring myself to write a react frontend for this on my day off).

One weird trick is if you run this with only the 1-bit quantized index without the re-ranking. The whole thing runs in 0.3 seconds, and the rankings are **not** that different.

```
$ time ./search.js --fast-index-only <<< 'Ice cream is poisoned'
Found 10 items
 1. (65.00) Lawyers question source of ice cream poisoning
 2. (65.00) Toxic 2
 3. (64.00) Man seeks compo over alleged ice cream poisoning
 4. (62.00) Toxic blanket
 5. (61.00) One year on from toxic orica spill
 6. (61.00) Leucaena toxicity
 7. (61.00) Womens deaths linked to antibiotics containing rat poison toxin
 8. (56.00) Toxic food
 9. (55.00) Poison thought to be behind wildlife deaths
10. (52.00) Poison warning
./search.js --fast-index-only <<< 'Ice cream is poisoned'  0.24s user 0.12s system 120% cpu 0.304 total
```

Could we get away with a 256-bit embedding and still provide useful results? I'm not sure. If I had more caffeine in my body I might start running the 256-bit embeddings against the standard benchmarks but I think this is a good place to stop and rethink our life choices.

We got there in the end. We can build our own index and run semantic search over a large-ish set using real portable JavaScript without any native extensions.

It's **medium** accurate and **medium** fast. I'm not sure if it's medium practical but we learned some things and most importantly I found a better embedding model for `ruru` (oops).

You can get:
1. the code from [github.internet](https://github.com/gerardpaapu/potion-js)
2. the data from [hugging-face](https://huggingface.co/datasets/rajistics/million-headlines/blob/main/abcnews-date-text_train.csv)
3. the determination to do something this stupid from somewhere deep inside you