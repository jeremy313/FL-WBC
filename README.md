# FL-WBC: Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective
Official implementation of "FL-WBC: Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective"


## Abstract

Federated learning (FL) is a popular distributed learning framework that trains a global model through iterative communications between a central server and edge devices. 
Recent works have demonstrated that FL is vulnerable to model poisoning attacks. Several server-based defense approaches (e.g. robust aggregation), have been proposed to mitigate such attacks. 
However, we empirically show that under extremely strong attacks, these defensive methods fail to guarantee the robustness of FL.
More importantly, we observe that as long as the global model is polluted, the impact of attacks on the global model will remain in subsequent rounds even if there are no subsequent attacks. 
In this work, we propose a client-based defense, named White Blood Cell for Federated Learning (FL-WBC), which can mitigate model poisoning attacks that have already polluted the global model. 
The key idea of FL-WBC is to identify the parameter space where long-lasting attack effect on parameters resides and perturb that space during local training. 
Furthermore, we can derive a certified robustness guarantee against model poisoning attacks and a convergence guarantee to FedAvg after applying our FL-WBC. 
We conduct experiments on FasionMNIST and CIFAR10 to evaluate the defense against state-of-the-art model poisoning attacks. 
The results demonstrate that our method can effectively mitigate model poisoning attack impact on the global model within 5 communication rounds with nearly no accuracy drop under both IID and Non-IID settings. 
Our defense is also complementary to existing server-based robust aggregation approaches and can further improve the robustness of FL under extremely strong attacks.


## Setup
```
pytorch=1.2.0
torchvision=0.4.0
```

## Quick start

For CIFAR10 dataset, you can reproduce the results of single image defense in the paper by running
```
python fedavg.py --dataset=cifar --num_users=100 --iid=1 --gpu=0 --frac=0.1 --model=cnn --epoch=500 --mal_boost=5 --local_mal_ep=10 --pert_strength=0.4 --num_mal_samples=1 --defense=WBC
```

## Important hyperparameters
```
--pert_strength: "s" in the paper (std of $\Upsilon$)
--mal_boost: the number of attackers in one round
```
