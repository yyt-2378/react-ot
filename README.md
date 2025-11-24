# React-OT: Optimal Transport for Generating Transition State in Chemical Reactions

In this work, we developed React-OT, an optimal transport approach to generate TSs of an elementary reaction in a fully deterministic manner. It is based on our previously developed diffusion-based generative model for generating 3D chemical reactions, [OA-ReactDiff](https://github.com/chenruduan/OAReactDiff). React-OT has been improved for generating transition state (TS) structures for a given reactants and products (double-ended search problem), enabling it to generate highly accurate transition state structures while maintaining an extremely high inference speed.

![image](https://github.com/deepprinciple/react-ot/blob/main/reactot/Figures/figure1.jpg)
Fig. 1 | Overview of the diffusion model and optimal transport framework for generating TS. a. Learning the joint distribution of structures in elementary reactions (reactant in red, TS in yellow, and product in blue). b. Stochastic inference with inpainting in OA-ReactDiff. c. Deterministic inference with React-OT.

We trained React-OT on Transition1x, a dataset that contains paired reactants, TSs, and products calculated from climbing-image NEB obtained with DFT (ωB97x/6-31G(d)). In React-OT, the object-aware version of LEFTNet is used as the scoring network to fit the transition kernel (see [LEFTNet](https://arxiv.org/abs/2304.04757)). React-OT achieves a mean RMSD of 0.103 Å between generated and true TS structures on the set-aside test reactions of Transition1x, significantly improved upon previous state-of-the-art TS prediction methods.

![image](https://github.com/deepprinciple/react-ot/blob/main/reactot/Figures/figure2.jpg)
Fig. 2 | Structural and energetic performance of diffusion and optimal transport generated TS structures. a. Cumulative probability for structure root mean square deviation (RMSD) (left) and absolute energy error (|∆E TS|) (right) between the true and generated TS on 1,073 set-aside test reactions.  b. Reference TS structure, OA-ReactDiff TS sample (red), and React-OT structure (orange) for select reactions. c. Histogram (gray, left y axis) and cumulative probability(blue, right y axis) showing the difference of RMSD (left) and |∆ETS|(right) between OA-ReactDiff recommended and React-OT structures compared to reference TS. d. Inference time in seconds for single-shot OA-ReactDiff, 40-shot OA-ReactDiff with recommender, and React-OT.

We envision that the remarkable accuracy and rapid inference of React-OT will be highly useful when integrated with the current high-throughput TS search workflow. This integration will facilitate the exploration of chemical reactions with unknown mechanisms.

## Environment set-up
```
conda env create -f env.yaml
conda activate reactot && pip install -e .
```

## Download data
The processed data is uploaded on zenodo, [download link](https://zenodo.org/records/13131875). You need to put both pickle files under the data directory.

```
mkdir reactot/data
mkdir reactot/data/transition1x
mv PATH_TO_PKL_FILES reactot/data/transition1x/
```

## Evaluation using a pre-trained model
The pre-trained model can be downloaded through the [download link](https://zenodo.org/records/13131875).
```
python evaluation.py --checkpoint PATH_TO_CHECKPOINT --solver ode --nfe 10
``` 

## Training
```
python -m reactot.trainer.train_rpsb_ts1x
```
Note that the default parameters and model types are used in the current command. More detailed instructions on model training will be updated soon.

## 用更口语的方式理解模型和训练流程
- **LEFTNet 像“大脑”**：它是一个等变图神经网络，负责看懂反应物/产物/TS 的几何和原子类型，并输出坐标/特征的更新。无论训练 DDPM 还是 SB，都复用同一套 LEFTNet 权重。
- **DDPM 像“先学会去噪的练兵场”**：给真实分子图逐步加噪，再让模型学会去噪还原。这样模型先学会“从混乱里复原分子”，可当作基线生成器，也可以预训练出一套不错的权重，后面再拿来做 SB（如果在配置里开启 `use_pretrain=True`）。
- **SB（Schrödinger Bridge）像“有方向的桥梁”**：直接把起点（反应物+产物的合成态）和终点（TS）作为两端，让模型学习一条最省事的桥接轨迹。推理时只给反应物+产物，就能沿着学到的桥一步到位生成 TS，通常比纯 DDPM 更贴近真实反应路径。
- **训练时怎么选**：当前默认配置是“直接训练 SB”（`use_pretrain=False`），不走 DDPM 预训练；如果显存或数据有限，可以先训练 DDPM 再把权重加载给 SB，收敛更稳。
- **怎么用**：
  - 想评测/推理：下载预训练权重，运行 `python evaluation.py --checkpoint PATH --solver ode --nfe 10`，给定反应物+产物就能生成 TS。
  - 想自己训练：准备好 Transition1x 数据后运行上面的训练命令；需要暖启动时改成 `use_pretrain=True` 并指定 DDPM 检查点，即可先加载去噪模型再继续 SB 训练。
- **一句话总结训练顺序**：
  1. 最快上手：直接训练 SB → 得到生成 TS 的模型（默认配置）。
  2. 更稳妥：先训练 DDPM 让 LEFTNet 学会“去噪/理解结构” → 把 DDPM 权重作为初始化加载到 SB → SB 再学习“如何从 R+P 走到 TS 的桥”。
  3. 无论走哪个流程，推理时都是用最终的 SB 检查点，输入反应物+产物，直接产出 TS。

### 输入 / 输出 怎么走（口语版）
- **模型看到的输入是什么？**
  - 训练 SB 时，批数据里带有 *三个* 图：反应物 (R)、过渡态 (TS)、产物 (P)。代码会把 R 和 P 的坐标/原子类型合并成“起点态”图（中间坐标常用 R/P 的几何平均），再与目标 TS 图一起送入 LEFTNet。
  - 如果先做 DDPM 预训练，则只有 TS 图本身：先加噪声再让 LEFTNet 去噪。
- **LEFTNet 输出什么？**
  - 每一前向都会输出：节点的隐藏特征（标量）以及坐标增量 `dpos`（矢量）。
  - 在 SB 里，`dpos` 会被当作“桥接漂移/速度场”，ODE 积分多步后把起点态的坐标推进到模型预测的 TS 坐标；节点特征也会参与能量/对齐等损失。
  - 在 DDPM 里，`dpos` 则对应“去噪残差”，用来还原被加噪的 TS 坐标，损失是“你是否把噪声去掉”。
- **推理/生成时的输出**：
  - 只需要输入反应物+产物（同训练时的起点构造），不用提供 TS。SB 会在 ODE 轨迹末端直接给出一套 3D 坐标，作为预测的 TS。整个过程是确定性的（给定相同超参会得到相同 TS）。

## LEFTNet 架构分步讲解（白话版）
### 架构框架示意图
```mermaid
flowchart LR
  subgraph Inputs
    R[反应物坐标/原子类型]
    P[产物坐标/原子类型]
    TS[TS 坐标\n（仅 SB 训练需要）]
  end

  subgraph Encoders
    RBF[RBFEmb 距离嵌入]
    NEB[NeighborEmb 邻域特征]
    Frame[局部坐标系构造]
    S2V[CFConvS2V 标量→矢量]
  end

  subgraph MessagePass[等变消息传递 (重复 num_layers 次)]
    GCL[GCLMessage 标量聚合]
    EquiMsg[EquiMessage 标量+矢量消息]
    Update[EquiUpdate 残差融合]
  end

  subgraph Outputs
    DPos[EquiOutput 坐标增量 dpos\n(桥接漂移/去噪残差)]
    Emb[embedding_out 节点隐藏特征]
  end

  R & P -->|构造起点图| RBF
  TS -->|SB 目标| Frame
  RBF --> NEB --> Frame --> S2V --> GCL --> EquiMsg --> Update
  Update -->|重复 L 层| GCL
  Update --> DPos
  Update --> Emb
```

> 下面按“输入 → 编码 → 消息传递 → 坐标更新 → 输出”的顺序，把 `reactot/model/leftnet.py` 里的关键模块串起来。你可以在文件里搜索相应类名快速定位。

1. **输入与距离编码：RBFEmb / NeighborEmb**  
   - 读取原子种类和 3D 坐标，先用径向基函数 `RBFEmb` 把原子间距离嵌入成一组平滑的标量特征；同时用 `NeighborEmb` 把原子类型信息扩展到邻居（类似“先看周围是什么原子”）。
2. **构建局部坐标系：scalarization + frame**  
   - 每条边都会构造一个三维正交基（由原子间向量、叉乘向量等组成），让后续的矢量特征都在统一的“本地坐标”下计算，从而保持旋转/反射等变性。
3. **标量 → 矢量映射：CFConvS2V**  
   - 利用局部帧，把前面得到的标量节点特征转换成方向相关的矢量特征，相当于告诉模型“这个信息指向哪儿”。
4. **等变消息传递：GCLMessage + EquiMessage**  
   - `GCLMessage` 先做一次标量域的消息聚合；`EquiMessage` 随后在标量+矢量的联合空间上传递信息并保持等变性。它用径向嵌入和注意力样式的权重，计算“邻居对当前原子施加的方向性更新”。
5. **等变更新与残差：EquiUpdate**  
   - 每层都会把新的标量/矢量特征做残差累加，并在需要时用 `EquiUpdate` 把矢量和标量重新融合（包含简单的“点积 + 小 MLP”），既稳住数值也增强表达力。
6. **多层叠加与动态系数**  
   - 上述消息传递/更新会重复若干层（由 `num_layers` 控制），层与层之间共享同样的等变设计；如果开启 `pos_grad`，还会用一个小 MLP 预测“动态系数”来微调坐标梯度。
7. **坐标与特征输出：EquiOutput / embedding_out**  
   - 通过 `EquiOutput` 把矢量特征转成坐标增量 `dpos`，再把标量特征送入 `embedding_out` 得到最终的节点隐藏表示。训练时，DDPM 或 SB 会用这些更新后的坐标/特征去计算对应的去噪或桥接损失。

一句话总结：LEFTNet 先把原子关系编码成等变的标量+矢量特征，再通过多层“看邻居 → 融合 → 残差”不断更新，最后输出新的坐标和隐藏特征，供 DDPM/SB 各自的训练目标使用。

## Data used in this work
1. [Transition1x](https://gitlab.com/matschreiner/Transition1x)
2. [RGD1](https://figshare.com/articles/dataset/model_reaction_database/21066901)
3. [Berkholz-15](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23910)
