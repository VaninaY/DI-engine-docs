Distributed Reinforcement Learning
===============

Problem Definition and Research Motivation
-------------------
Distributed reinforcement learning (Distributed RL) is the only way for deep reinforcement learning to be applied to large-scale applications and solve complex decision spaces and long-term planning problems. 为了解决像星际争霸2（SC2） [1]_ 和 DOTA2 [2]_ 这样超大规模的决策问题，单进程乃至单机器的算力是远远不够的，需要将整个训练管线中的各个部分拓展到各种各样的计算和存储设备上。研究者们希望设计一整套“算法+系统”的方案，能够让 DRL 训练程序便捷地运行在各种不同的计算尺度下，在保证算法优化收敛性的同时，尽可能地提升其中各个环节的效率。

一般来说，一个强化学习训练程序有三类核心模块，用于和环境交互产生数据的 Collector，其中包含环境本身（Env）和产生动作的 Actor，以及使用这些数据进行训练的 Learner，他们各自需要不同数量和类型的计算资源支持。而根据算法和环境类型的不同，又会有一些延伸的辅助模块，例如大部分 off-policy 算法都会需要数据队列（Replay Buffer）来存储训练数据，对于 model-based RL 相关的算法又会有学习环境 dynamics 的相关训练模块，而对于需要大量自我博弈（self-play）的算法，还需要一个中心化的 Coordinator 去控制协调各个组件（例如动态指定自己博弈的双方）。

在系统角度，需要让整个训练程序中的同类模块有足够的并行扩展性，例如可以根据需求增加进行交互的环境数量（消耗更多的CPU），或是增加训练端的吞吐量（通用需要使用更多的GPU），而对于不同的模块，又希望能够尽可能地让所有的模块可以异步执行，并减小模块时间各种通信方式的代价（网络通信，数据库，文件系统）。但总的来说，一个系统的效率优化的理论上限是——Learner 无等待持续高效训练，即在 Learner 一个训练迭代高效完成时，下一个训练迭代的数据已经准备好。

在算法角度，则是希望在保证算法收敛性的情况下，降低算法对数据产生吞吐量的要求（例如容忍更旧更 off-policy 的数据），提高数据探索效率和对于已收集数据的利用效率（例如修改数据采样方法，或是结合一些 RL 中 data-efficiency 相关的研究），从而为系统设计提供更大的空间和可能性。

综上所述，分布式强化学习是一个更加综合的研究子领域，需要深度强化学习算法 + 分布式系统设计的互相感知和协同。


Research Direction
---------

System
~~~~~~

Overall Architecture
^^^^^^^^^^
For common decision problems, the two most commonly used distributed architectures are IMPALA [3]_ and SEED RL [4]_

.. image:: ./images/impala.png
  :align: center
  
- The former is the classic Actor-Learner mode, that is, the data collection and training sides are completely separated, and the latest neural network model is regularly passed from the Learner to the Actor, and the Actor is sent to the Learner after collecting a certain amount of data (i.e. observations). If there are multiple Learners, they also periodically synchronize the gradients of the neural network (i.e. the data-parallel mode in distributed deep learning).

.. image:: ./images/seed_rl.png
  :scale: 50%
  :align: center

- On the basis of the former, the latter is dedicated to optimizing the loss of the transmission model. SEED RL strips out the part used for inference to generate actions, and puts it together with the training end to update the model through efficient inter-TPU communication technology. The cost of passing the model in IMPALA is greatly reduced, and for cross-machine communication between the environment and the reasoning Actor, SEED RL uses an optimized gRPC scheme to pass the observation and action, so there is not much burden.

.. note::
There is no absolute superiority or inferiority between these two schemes. The key lies in the fact that for a practical decision-making problem, whether it is more expensive to transmit models across machines, or more expensive to transmit observation and action data across machines, if it is the former , and there are better communication components between GPU/TPU, then SEED RL is a better solution, if it is the latter, IMPALA is a more stable choice. In addition, IMPALA can accumulate a batch of data for data transmission, while SEED RL requires data transmission in each interactive frame. This is a classic data batch and stream processing comparison problem. For the current machine learning community, the former is generally more complex. Ease of use. Also, if the entire training procedure requires a higher degree of freedom and customization, such as dynamically controlling some behavior of the Actor, IMPALA is more convenient.

In addition to the above two architectures, there are many other distributed reinforcement learning design schemes, such as A3C [5]_ and Gossip A2C [6]_ that introduce asynchronous neural network update schemes, In order to support large-scale self-play, AlphaStar [1]_ with a complex League mechanism was designed, and MuZero [7]_ combined with model-based RL and MCTS-related modules will not be described here. Interested readers can refer to the specific Papers or refer to our `Algorithm Raiders Collection section <../12_policies/index_zh.html>`_.

Single Point Efficiency Optimization
^^^^^^^^^^^^^
In addition to the design and innovation of the overall structure, there are many methods for optimizing a single-point module in the entire training program. They are mainly customized and optimized for a certain sub-problem. Here are some main methods:

- ``Object Store`` in Ray/RLLib [8]_: For data transfer between multiple processes and multiple machines, the Object Store in Ray/RLLib provides a very convenient and efficient way. As long as any process knows the reference of this object (that is, the reference), it can request the Store by requesting it. Obtain the corresponding value, and the specific internal data transmission is completely managed by the Store, so that a distributed training program can be implemented like writing a local single-process program. The specific implementation of Object Store is completed by combining redis, plasma and gRPC.

- ``Sample Factory`` [9]_: Sample Factory has customized and optimized the APPO algorithm at the scale of a single machine, carefully designed an asynchronous scheme between the environment and the action-generating strategy, and used shared memory to greatly improve the transmission efficiency between modules.

- ``Reverb`` in Acme [10]_: Reverb provides a set of highly flexible and efficient data manipulation and management modules. For RL, it is very suitable for implementing replay buffer related components.

- ``envpool`` [11]_: envpool is currently the fastest environment vectorized parallel solution, using c++ threadpool and efficient implementation of many classic RL environments to provide powerful asynchronous vectorized environment simulation capabilities.


Algorithm
~~~~~~

Reduce the throughput requirements of the algorithm for data generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``V-trace`` in IMPALA [3]_: The off-policy algorithm can widen the range of data available for training, thereby improving the algorithm's tolerance for old data to a certain extent and reducing the throughput pressure of the data generated by the Collector, but the data that is too off-policy can easily affect the convergence of the algorithm. Aiming at this problem, IMPALA uses the importance sampling mechanism and the corresponding clipping method to design a relatively stable algorithm scheme V-trace under the distributed training setting, which limits the negative impact of off-policy data on the optimization itself.

- ``Reuse`` and ``Staleness`` in OpenAI FIVE [2]_: In the agent designed by OpenAI for DOTA2, they conducted some experiments on the number of data reuse (Reuse) and the degree of depreciation (Staleness). Excessive number of reuse and too old data will affect the stability of the PPO algorithm in large-scale training.

Improve data exploration efficiency + utilization efficiency of collected data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``Data Priority and Diversity``——Ape-x [12]_: Ape-x is a classic distributed reinforcement learning scheme. One of the core practices is to use Priority Experience Replay to set different sampling priorities for different data, so that the algorithm pays more attention to those "important" data. In addition, Ape-x also sets different exploration parameters (ie epsilon of eps greedy) in different parallel collectors to improve data diversity.

- ``Representation Learning`` in RL——CURL [13]_: For some high-dimensional or multi-modal inputs, the representation learning method can be combined to improve the data utilization efficiency of RL. For example, for the control problem of high-dimensional image input, CURL introduces an additional contrastive learning process, and RL is based on the learned feature space for decision-making. From the perspective of system design, there is also a lot of room for optimization in the combination of representation learning and reinforcement learning training, such as the asynchrony of the two.

- ``Model-based/MCTS RL``——MuZero [7]_: MuZero combines model-based RL and MCTS RL to improve the overall training efficiency, which includes many unique modules, such as the search process of MCTS, the reanalyze process of data before training, etc., which will lead to more complicated and diverse distributed reinforcement learning training systems.

Future Study
---------

At present, distributed reinforcement learning is only an emerging research subfield. In many cases, it is limited by computing power and problem environment. There are still many problems that need to be solved:

- Lack of a unified benchmark to evaluate the efficiency of distributed reinforcement learning algorithms and systems;

- At present, most distributed reinforcement learning solutions are only suitable for a small part of the environment and part of the RL algorithm, and there is still a long way to go before the generalization of the technology;

- Current system optimization and RL algorithms themselves are still isolated, and system designs that sense RL optimization needs can be considered, such as dynamic resource awareness and scheduling.


Reference
----------
.. [1] Oriol Vinyals, Igor Babuschkin, David Silver, et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nat. 575(7782): 350-354 (2019)

.. [2] Christopher Berner, Greg Brockman, et al. Dota 2 with Large Scale Deep Reinforcement Learning. CoRR abs/1912.06680 (2019)

.. [3] Lasse Espeholt, Hubert Soyer, Rémi Munos, et al. IMPALA. Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. ICML 2018: 1406-1415

.. [4] Lasse Espeholt, Raphaël Marinier, Piotr Stanczyk, Ke Wang, Marcin Michalski. SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference. ICLR 2020

.. [5] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. ICML 2016: 1928-1937

.. [6] Mahmoud Assran, Joshua Romoff, Nicolas Ballas, Joelle Pineau, Mike Rabbat. Gossip-based Actor-Learner Architectures for Deep Reinforcement Learning. NeurIPS 2019: 13299-13309

.. [7] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy P. Lillicrap, David Silver. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. CoRR abs/1911.08265 (2019)

.. [8] Eric Liang, Richard Liaw, Robert Nishihara, Philipp Moritz, Roy Fox, Joseph Gonzalez, Ken Goldberg, Ion Stoica. Ray RLLib: A Composable and Scalable Reinforcement Learning Library. CoRR abs/1712.09381 (2017)

.. [9] Aleksei Petrenko, Zhehui Huang, Tushar Kumar, Gaurav S. Sukhatme, Vladlen Koltun. Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning. ICML 2020: 7652-7662

.. [10] Matt Hoffman, Bobak Shahriari, John Aslanides, Gabriel Barth-Maron, Feryal Behbahani, Tamara Norman, Abbas Abdolmaleki, Albin Cassirer, Fan Yang, Kate Baumli, Sarah Henderson, Alexander Novikov, Sergio Gómez Colmenarejo, Serkan Cabi, Çaglar Gülçehre, Tom Le Paine, Andrew Cowie, Ziyu Wang, Bilal Piot, Nando de Freitas. Acme: A Research Framework for Distributed Reinforcement Learning. CoRR abs/2006.00979 (2020)

.. [11] Jiayi Weng and Min Lin and Zhongwen Xu and Shuicheng Yan. https://github.com/sail-sg/envpool


.. [12] Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van Hasselt, David Silver. Distributed Prioritized Experience Replay. ICLR (Poster) 2018

.. [13] Michael Laskin, Aravind Srinivas, Pieter Abbeel: CURL: Contrastive Unsupervised Representations for Reinforcement Learning. ICML 2020: 5639-5650
