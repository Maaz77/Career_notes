# ViT

- At the end of the decoder, every patch finds its meaning a long the whole image.
- if we input the encoder with n tokens, the output will also be n tokens.
- we add a CLS token to input of encoder, and at the end it will represent the whole image.
- in Vit, we have a global receptive field for every patch. 

## Attention
- QUERY is the what the patch is looking for in the image.
- KEY is the what the patch has to offer. (Like the headline of an article)
- VALUE is the actual information of the patch. (Like the content of the article)
- Using the QUERY and the KEY we calculate the attention score. Then we weight sum the VALUEs of other patches with the attention score to get the final value for the patch.

### Drawbacks of ViT

- Fixed input size because of trainable position embedding.
- Quadratic complexity of self-attention.
- Data hungry: needs large datasets to perform well due to multi head attention > inductive bias = the reason your model can generalize at all.
    - Solution: Token-to-Token ViT (T2T-ViT) which recursively aggregates neighboring tokens into one token, thus reducing the sequence length and increasing the receptive field of each token.
    - Solution: DeIT (Data-efficient image Transformers) which uses knowledge distillation from a CNN teacher to improve performance on smaller datasets.

- Lack of hierarchical structure, which is good thing for multiscale objects detection.
    - Solution: Pyramid ViT (PVT) which introduces a pyramid structure to capture multi-scale features by progressively reducing the spatial resolution of the feature maps.
    

