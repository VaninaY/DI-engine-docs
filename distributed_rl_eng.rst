Distributed Reinforcement Learning
===============

Problem Definition and Research Motivation
-------------------
Distributed reinforcement learning (Distributed RL) is the only way for deep reinforcement learning to be applied to large-scale applications and solve complex decision spaces and long-term planning problems. In order to solve super-large-scale decision-making problems like StarCraft 2 (SC2) [1]_ and DOTA2 [2]_, the computing power of a single process or even a single machine is far from enough, and it is necessary to expand each part of the entire training pipeline and to a wide variety of computing and storage devices. The researchers hope to design a complete set of "algorithm + system" solutions, which can allow the DRL training program to efficiently run under various computing scales and improve the efficiency of each link as much as possible while ensuring the convergence of algorithm optimization. 

Generally speaking, a reinforcement learning training program has three types of core modules, the Collector for interacting with the Environment (Env) to generate data, which contains the environment itself and the Actor that generates actions, and the Learner for training using these data, each of which requires different number and types of computing resources supported.

Depending on the algorithm and environment, some extended auxiliary modules will be added. For example, most off-policy algorithms will require a data queue (Replay Buffer) to store training data, and there will be learning for model-based RL-related algorithms. For the relevant training modules of environmental dynamics and algorithms that require a large number of self-play (self-play), a centralized Coordinator is also required to control and coordinate various components (such as dynamically specifying both sides of the game).

From a system perspective, it is necessary to allow sufficient parallel scalability for similar modules in the entire training program. For example, the number of interacting environments can be increased according to demand (consume more CPU), or the throughput of the training side can be increased (generally, more more GPUs). For different modules, it is hoped that all modules can be executed asynchronously as much as possible, and the cost of various communication methods (network communication, database, file system) in module time is reduced. But in general, the theoretical upper limit of the efficiency optimization of a system is that the Learner can continuously train efficiently without waiting; that is, when the Learner completes one training iteration efficiently, the data for the next training iteration is ready.

From the algorithm point of view, it is hoped to reduce the algorithm’s requirements for data throughput (such as tolerating older and more off-policy data) while ensuring the convergence of the algorithm, and improving the efficiency of data exploration and utilization of collected data (For example, modify the data sampling method, or combine some research related to data-efficiency in RL). So that it provides more space and possibilities for system design.

To sum up, distributed reinforcement learning is a more comprehensive research subfield, which requires mutual perception and coordination of deep reinforcement learning algorithm + distributed system design.


Research Direction
---------

System
~~~~~~

Overall Architecture
^^^^^^^^^^
For common decision problems, the two most commonly used distributed architectures are IMPALA [3]_ and SEED RL [4]_

.. image:: ./images/impala.png
  :align: center
  
- The former is the classic Actor-Learner mode; that is, the data collection and training sides are entirely separated, and the latest neural network model is regularly passed from the Learner to the Actor, and the Actor is sent to the Learner after collecting a certain amount of data (i.e. observations). If there are multiple Learners, they also periodically synchronize the gradients of the neural network (i.e. the data-parallel model in distributed deep learning).

.. image:: ./images/seed_rl.png
  :scale: 50%
  :align: center

- On the basis of the former, the latter is dedicated to optimizing the loss of the transmission model. SEED RL strips out the part used for inference to generate actions, and puts it together with the training end to update the model through efficient inter-TPU communication technology. The cost of passing the model in IMPALA is greatly reduced, and for cross-machine communication between the environment and the reasoning Actor, SEED RL uses an optimized gRPC scheme to pass the observation and action, so there is not much burden.

.. note::
There is no absolute superiority or inferiority between these two schemes. The key lies in the fact that for a practical decision-making problem, whether it is more expensive to transmit models across machines, or more expensive to transmit observation and action data across machines, if it is the former , and there are better communication components between GPU/TPU, then SEED RL is a better solution, if it is the latter, IMPALA is a more stable choice. In addition, IMPALA can accumulate a batch of data for data transmission, while SEED RL requires data transmission in each interactive frame. This is a classic data batch and stream processing comparison problem. For the current machine learning community, the former is generally more complex and user-friendly. Also, if the entire training procedure requires a higher degree of freedom and customization, such as dynamically controlling some behavior of the Actor, IMPALA is more convenient.

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
