# InfoMatch: Entropy Neural Estimation for Semi-Supervised Image Classification
Semi-supervised image classification, leveraging pseudo supervision and consistency regularization, has demonstrated remarkable success. However, the ongoing challenge lies in fully exploiting the potential of unlabeled data. To address this, we employ information entropy neural estimation to harness the potential of unlabeled samples. Inspired by contrastive learning, the entropy is estimated by maximizing a lower bound on mutual information across different augmented views. Moreover, we theoretically analyze that the information entropy of the posterior of an image classifier is approximated by maximizing the likelihood function of the softmax predictions. Guided by these insights, we optimize our model from both perspectives to ensure that the predicted probability distribution closely aligns with the ground-truth distribution. Given the theoretical connection to information entropy, we name our method **InfoMatch**. Through extensive experiments, we show its superior performance.


# Citation
We appreciate it if you cite the following paper:
```
@InProceedings{Hanijcai2024,
  author =    {Qi Han and Zhibo Tian and Chengwei Xia and Kun Zhan},
  title =     {InfoMatch: Entropy neural estimation for semi-supervised image classification},
  booktitle = {IJCAI},
  year =      {2024},
  volume =    {33},
}
```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)