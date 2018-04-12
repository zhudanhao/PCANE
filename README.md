# NAP_ANE

This is the work for paper 'Preserving Neighborhoods’ Attributes for Network Embedding'

Danhao Zhu1;2, Xin-yu Dai1, and Kaijia Yang1
1 Nanjing University, Nanjing,220000, China
fzhudh,yangkjg@nlp.nju.edu.cn,daixinyu@nju.edu.cn
2 Jiangsu Police Institute, Nanjing, 220000, China



all the data is prepocessed and indexed in the 'data' fold



Here is the abstract.

Abstract. Mapping network nodes to low-dimensional vectors has shown
promising results for many downstream tasks, such as link prediction and
node classification. Recently, researchers found integrating attributes will
improve the quality of learned vectors. However, for each represented node, the existing methods can only considered its own attributes, but
ignored the attributes of its neighborhoods, which are also indicative to
the node’s semantic.
To properly utilize the information, we propose a new algorithm named
NAP ANE(Neighborhoods’ Attributes Proximity for Attributed Network
Embedding) to learn vector representation for nodes of attributed network. Compared to previous works, our method can capture not only the
network structure, but also the neighborhoods’ attributes of network, and
hence produces more informative node representations.
Extensive experiments on four real-world attributed networks show that
compared to state-of-the-art baselines, our method learns better representations and achieves substantial performance gains on link prediction
and node classification tasks.
