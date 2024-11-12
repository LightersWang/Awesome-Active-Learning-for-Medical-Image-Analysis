# Awesome Active Learning for Medical Image Analysis

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![arXiv](https://img.shields.io/badge/arXiv-2310.14230-b31b1b.svg)](https://arxiv.org/abs/2310.14230)

**🎉🎉🎉 Our survey is accepted by Medical Image Analysis (IF = 10.9) !**

In this repo, we provide a paper list of active learning in the fields of medical image analysis and computer vision. We focus on active learning papers of medical image analysis which were published on top-tier journals or conferences.

Also, we provide the code to evaluate different active learning methods on multiple medical imaging datasets of classification or segmentation. Please refer to the `code` folder.

This repo is mainly based on the following survey:  
**A comprehensive survey on deep active learning in medical image analysis**  
_Haoran Wang, Qiuye Jin, Shiman Li, Siyu Liu, Manning Wang*, Zhijian Song*_  
_\* -- Corresponding Author_  
[Paper](https://www.sciencedirect.com/science/article/pii/S1361841524001269) | [arXiv](https://arxiv.org/abs/2310.14230)

If you find this repo or this survey paper helpful, please consider citing:

```bibtex
@article{wang2024comprehensive,
  title={A comprehensive survey on deep active learning in medical image analysis},
  author={Wang, Haoran and Jin, Qiuye and Li, Shiman and Liu, Siyu and Wang, Manning and Song, Zhijian},
  journal={Medical Image Analysis},
  pages={103201},
  year={2024},
  publisher={Elsevier}
}
```

Currently, I am actively updating this repo. Newly published papers and missing papers of related topics will be added. **Please star us if you find this repo helpful ⭐**

**If you have any questions or suggestions, or if you think your paper fits our framework and should be added in the survey, feel free to contact me through e-mail (hrwang20@fudan.edu.cn) or create an issue in this repo.**

We also welcome contributions to this repo for updating the paper list. Please submit a pull request if you want to contribute!


## Paper List 

> **[[PDF]]()** - PDF link  
> **[[Code]]()** - official code link  
> 🕝 - not cited in our survey paper  

### Surveys

A comprehensive survey on deep active learning in medical image analysis  
**[2024 MedIA]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001269)**

Label-efficient deep learning in medical image analysis: Challenges and future directions  
**[2023 arXiv]** **[[PDF]](https://arxiv.org/pdf/2303.12484.pdf)**

Deep Active Learning for Computer Vision: Past and Future  
**[2023 APSIPA Transactions on Signal and Information Processing]** **[[PDF]](https://arxiv.org/pdf/2211.14819.pdf)**

A comparative survey of deep active learning  
**[2022 arxiv]** **[[PDF]](https://arxiv.org/pdf/2203.13450.pdf)**

A survey on active deep learning: from model driven to data driven  
**[2022 ACM Computing Surveys]** **[[PDF]](https://dl.acm.org/doi/abs/10.1145/3510414)**

A survey on active learning and human-in-the-loop deep learning for medical image analysis  
**[2021 MedIA]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841521001080)**

A survey of deep active learning  
**[2021 ACM Computing Surveys]** **[[PDF]](https://dl.acm.org/doi/abs/10.1145/3472291)**

Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation  
**[2020 MedIA]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S136184152030058X)**

Active learning literature survey  
**[2009]** **[[PDF]](https://minds.wisconsin.edu/bitstream/handle/1793/60660/TR1648.pdf?sequence=1)**


### IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

ActiveDC: Distribution Calibration for Active Finetuning  
🕝 **[CVPR'24]** **[[PDF]](http://arxiv.org/abs/2311.07634)**

Active Generalized Category Discovery  
🕝 **[CVPR'24]** **[[PDF]](http://arxiv.org/abs/2403.04272)**

Revisiting the Domain Shift and Sample Uncertainty in Multi-source Active Domain Transfer  
🕝 **[CVPR'24]** **[[PDF]](http://arxiv.org/abs/2311.12905)**

Active Prompt Learning in Vision Language Models  
🕝 **[CVPR'24]** **[[PDF]](http://arxiv.org/abs/2311.11178)**

Plug and Play Active Learning for Object Detection  
🕝 **[CVPR'24]** **[[PDF]](http://arxiv.org/abs/2211.11612)**

Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts  
**[CVPR'24]** **[[PDF]](https://arxiv.org/abs/2312.02567)**

Re-Thinking Federated Active Learning Based on Inter-Class Diversity  
🕝 **[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Re-Thinking_Federated_Active_Learning_Based_on_Inter-Class_Diversity_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/raymin0223/LoGo)**

Hybrid Active Learning via Deep Clustering for Video Action Detection  
🕝 **[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Rana_Hybrid_Active_Learning_via_Deep_Clustering_for_Video_Action_Detection_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/aayushjr/HybridCLAUS)**

Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-Based Active Learning  
🕝 **[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Ji_Are_Binary_Annotations_Sufficient_Video_Moment_Retrieval_via_Hierarchical_Uncertainty-Based_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/renjie-liang/HUAL)**

MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation  
🕝 **[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MHPL_Minimum_Happy_Points_Learning_for_Active_Source_Free_Domain_CVPR_2023_paper.pdf)**

Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm  
**[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Active_Finetuning_Exploiting_Annotation_Budget_in_the_Pretraining-Finetuning_Paradigm_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/yichen928/ActiveFT)**

Bi3D: Bi-Domain Active Learning for Cross-Domain 3D Object Detection  
**[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yuan_Bi3D_Bi-Domain_Active_Learning_for_Cross-Domain_3D_Object_Detection_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/JiakangYuan/Bi3D)**

Divide and Adapt: Active Domain Adaptation via Customized Learning  
**[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Divide_and_Adapt_Active_Domain_Adaptation_via_Customized_Learning_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/Duojun-Huang/DiaNA-CVPR2023)**

Box-Level Active Detection  
**[CVPR'23]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Lyu_Box-Level_Active_Detection_CVPR_2023_paper.pdf)** **[[Code]](https://github.com/lyumengyao/blad)**

Entropy-Based Active Learning for Object Detection With Progressive Diversity Constraint  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Entropy-Based_Active_Learning_for_Object_Detection_With_Progressive_Diversity_Constraint_CVPR_2022_paper.pdf)**

Active Learning for Open-Set Annotation  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ning_Active_Learning_for_Open-Set_Annotation_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/PrateekMunjal/TorchAL)**

Meta Agent Teaming Active Learning for Pose Estimation  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gong_Meta_Agent_Teaming_Active_Learning_for_Pose_Estimation_CVPR_2022_paper.pdf)**

Towards Robust and Reproducible Active Learning Using Neural Networks  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/PrateekMunjal/TorchAL)**

Active Learning by Feature Mixing  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Parvaneh_Active_Learning_by_Feature_Mixing_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/AminParvaneh/alpha_mix_active_learning)**

Which Images To Label for Few-Shot Medical Landmark Detection?  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Quan_Which_Images_To_Label_for_Few-Shot_Medical_Landmark_Detection_CVPR_2022_paper.pdf)**

Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Towards_Fewer_Annotations_Active_Learning_via_Region_Impurity_and_Prediction_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/BIT-DA/RIPU)**

Learning Distinctive Margin Toward Active Domain Adaptation  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Learning_Distinctive_Margin_Toward_Active_Domain_Adaptation_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/TencentYoutuResearch/ActiveLearning-SDM)**

BoostMIS: Boosting Medical Image Semi-Supervised Learning With Adaptive Pseudo Labeling and Informative Active Annotation  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_BoostMIS_Boosting_Medical_Image_Semi-Supervised_Learning_With_Adaptive_Pseudo_Labeling_CVPR_2022_paper.pdf)** **[[Code]](https://github.com/wannature/BoostMIS)**

One-Bit Active Query With Contrastive Pairs  
**[CVPR'22]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_One-Bit_Active_Query_With_Contrastive_Pairs_CVPR_2022_paper.pdf)**

Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning)**

Sequential Graph Convolutional Network for Active Learning  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Caramalau_Sequential_Graph_Convolutional_Network_for_Active_Learning_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning)**

VaB-AL: Incorporating Class Imbalance and Difficulty With Variational Bayes for Active Learning  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_VaB-AL_Incorporating_Class_Imbalance_and_Difficulty_With_Variational_Bayes_for_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/jongwon20000/vabal)**

Transferable Query Selection for Active Domain Adaptation  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Transferable_Query_Selection_for_Active_Domain_Adaptation_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/thuml/Transferable-Query-Selection)**

Exploring Data-Efficient 3D Scene Understanding With Contrastive Scene Contexts  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Exploring_Data-Efficient_3D_Scene_Understanding_With_Contrastive_Scene_Contexts_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/facebookresearch/ContrastiveSceneContexts)**

Task-Aware Variational Adversarial Active Learning  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Task-Aware_Variational_Adversarial_Active_Learning_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/cubeyoung/TA-VAAL)**

Multiple Instance Active Learning for Object Detection  
**[CVPR'21]** **[[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yuan_Multiple_Instance_Active_Learning_for_Object_Detection_CVPR_2021_paper.pdf)** **[[Code]](https://github.com/yuantn/MI-AOD)**

Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision  
🕝 **[CVPR'20]** **[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gudovskiy_Deep_Active_Learning_for_Biased_Datasets_via_Fisher_Kernel_Self-Supervision_CVPR_2020_paper.pdf)** **[[Code]](https://github.com/gudovskiy/al-fk-self-supervision)**

State-Relabeling Adversarial Active Learning  
**[CVPR'20]** **[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_State-Relabeling_Adversarial_Active_Learning_CVPR_2020_paper.pdf)** **[[Code]](https://github.com/Beichen1996/SRAAL)**

ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation  
**[CVPR'20]** **[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.pdf)** **[[Code]](https://github.com/nihalsid/ViewAL)**

Learning Loss for Active Learning  
**[CVPR'19]** **[[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf)**

Reducing Uncertainty in Undersampled MRI Reconstruction with Active Acquisition  
**[CVPR'19]** **[[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.pdf)**

The Power of Ensembles for Active Learning in Image Classification  
**[CVPR'18]** **[[PDF]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)**

Quantization of Fully Convolutional Networks for Accurate Biomedical Image Segmentation  
**[CVPR'18]** **[[PDF]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Quantization_of_Fully_CVPR_2018_paper.pdf)**

Fine-Tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally  
**[CVPR'17]** **[[PDF]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Fine-Tuning_Convolutional_Neural_CVPR_2017_paper.pdf)** **[[Code]](https://github.com/MrGiovanni/Active-Learning)**


### International Conference on Computer Vision (ICCV)
HAL3D: Hierarchical Active Learning for Fine-Grained 3D Part Labeling  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_HAL3D_Hierarchical_Active_Learning_for_Fine-Grained_3D_Part_Labeling_ICCV_2023_paper.pdf)**

Hierarchical Point-based Active Learning for Semi-supervised Point Cloud Semantic Segmentation  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Hierarchical_Point-based_Active_Learning_for_Semi-supervised_Point_Cloud_Semantic_Segmentation_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/SmiletoE/HPAL)**

ALWOD: Active Learning for Weakly-Supervised Object Detection  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_ALWOD_Active_Learning_for_Weakly-Supervised_Object_Detection_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/seqam-lab/ALWOD)**

You Never Get a Second Chance To Make a Good First Impression: Seeding Active Learning for 3D Semantic Segmentation  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Samet_You_Never_Get_a_Second_Chance_To_Make_a_Good_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/nerminsamet/seedal)**

Heterogeneous Diversity Driven Active Learning for Multi-Object Tracking  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Heterogeneous_Diversity_Driven_Active_Learning_for_Multi-Object_Tracking_ICCV_2023_paper.pdf)**

TiDAL: Learning Training Dynamics for Active Learning  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kye_TiDAL_Learning_Training_Dynamics_for_Active_Learning_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/hyperconnect/TiDAL)**

Knowledge-Aware Federated Active Learning with Non-IID Data  
🕝 **[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Knowledge-Aware_Federated_Active_Learning_with_Non-IID_Data_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/ycao5602/KAFAL)**

Adaptive Superpixel for Active Learning in Semantic Segmentation  
**[ICCV'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Adaptive_Superpixel_for_Active_Learning_in_Semantic_Segmentation_ICCV_2023_paper.pdf)** **[[Code]](https://github.com/ml-postech/adaptive-superpixel-for-active-learning-in-semantic-segmentation)**

Active Universal Domain Adaptation  
🕝 **[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Active_Universal_Domain_Adaptation_ICCV_2021_paper.pdf)**

Semi-Supervised Active Learning for Semi-Supervised Models: Exploit Adversarial Examples With Graph-Based Virtual Labels  
🕝 **[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Semi-Supervised_Active_Learning_for_Semi-Supervised_Models_Exploit_Adversarial_Examples_With_ICCV_2021_paper.pdf)**

Active Learning for Deep Object Detection via Probabilistic Modeling  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/NVlabs/AL-MDN)**

Contrastive Coding for Active Learning Under Class Distribution Mismatch  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Du_Contrastive_Coding_for_Active_Learning_Under_Class_Distribution_Mismatch_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/RUC-DWBI-ML/CCAL)**

Semi-Supervised Active Learning With Temporal Output Discrepancy  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Semi-Supervised_Active_Learning_With_Temporal_Output_Discrepancy_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/siyuhuang/TOD)**

Influence Selection for Active Learning  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Influence_Selection_for_Active_Learning_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/dragonlzm/ISAL)**

Multi-Anchor Active Domain Adaptation for Semantic Segmentation  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ning_Multi-Anchor_Active_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/munanning/MADA)**

Active Learning for Lane Detection: A Knowledge Distillation Approach  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Active_Learning_for_Lane_Detection_A_Knowledge_Distillation_Approach_ICCV_2021_paper.pdf)**

Active Domain Adaptation via Clustering Uncertainty-Weighted Embeddings  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/virajprabhu/CLUE)**

S3VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Rangwani_S3VAADA_Submodular_Subset_Selection_for_Virtual_Adversarial_Active_Domain_Adaptation_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/val-iisc/s3vaada)**

LabOR: Labeling Only If Required for Domain Adaptive Semantic Segmentation  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Shin_LabOR_Labeling_Only_if_Required_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf)**

ReDAL: Region-Based and Diversity-Aware Active Learning for Point Cloud Semantic Segmentation  
**[ICCV'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_ReDAL_Region-Based_and_Diversity-Aware_Active_Learning_for_Point_Cloud_Semantic_ICCV_2021_paper.pdf)** **[[Code]](https://github.com/tsunghan-wu/ReDAL)**

Active Learning for Deep Detection Neural Networks  
**[ICCV'19]** **[[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Aghdam_Active_Learning_for_Deep_Detection_Neural_Networks_ICCV_2019_paper.pdf)** **[[Code]](https://gitlab.com/haghdam/deep_active_learning)**

Deep Reinforcement Active Learning for Human-in-the-Loop Person Re-Identification  
**[ICCV'19]** **[[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Deep_Reinforcement_Active_Learning_for_Human-in-the-Loop_Person_Re-Identification_ICCV_2019_paper.pdf)**

Variational Adversarial Active Learning  
**[ICCV'19]** **[[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.pdf)** **[[Code]](https://github.com/sinhasam/vaal)**


### ICCV Workshop
Computational Evaluation of the Combination of Semi-Supervised and Active Learning for Histopathology Image Segmentation with Missing Annotations  
**[ICCVW'23]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Jimenez_Computational_Evaluation_of_the_Combination_of_Semi-Supervised_and_Active_Learning_ICCVW_2023_paper.pdf)**

Reducing Label Effort: Self-Supervised Meets Active Learning  
**[ICCVW'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Bengar_Reducing_Label_Effort_Self-Supervised_Meets_Active_Learning_ICCVW_2021_paper.pdf)**

Joint semi-supervised and active learning for segmentation of gigapixel pathology images with cost-effective labeling  
**[ICCVW'21]** **[[PDF]](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Lai_Joint_Semi-Supervised_and_Active_Learning_for_Segmentation_of_Gigapixel_Pathology_ICCVW_2021_paper.pdf)**


### European Conference on Computer Vision (ECCV)
Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00073.pdf)**

Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02596.pdf)**

Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04026.pdf)**

Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04970.pdf)**

Efficient Active Domain Adaptation for Semantic Segmentation by Selecting Information-rich Superpixels  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05060.pdf)**

Two-Stage Active Learning for Efficient Temporal Action Segmentation  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06348.pdf)**

Active Generation for Image Classification  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06459.pdf)**

Dataset Quantization with Active Learning based Adaptive Sampling  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07772.pdf)**

MetaAT: Active Testing for Label-Efficient Evaluation of Dense Recognition Tasks  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10062.pdf)**

Generalized Coverage for More Robust Low-Budget Active Learning  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11051.pdf)**

Exploring Active Learning in Meta-Learning: Enhancing Context Set Labeling  
🕝 **[ECCV'24]** **[[PDF]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12574.pdf)**

Optical Flow Training under Limited Label Budget via Active Learning  
🕝 **[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_24)** **[[Code]](https://github.com/duke-vision/optical-flow-active-learning-release)**

ActiveNeRF: Learning where to See with Uncertainty Estimation  
🕝 **[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_14)** **[[Code]](https://github.com/LeapLabTHU/ActiveNeRF)**

Active Label Correction Using Robust Parameter Update and Entropy Propagation  
🕝 **[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_1)**

LiDAL: Inter-Frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19812-0_15)** **[[Code]](https://github.com/hzykent/LiDAL)**

Combating Label Distribution Shift for Active Domain Adaptation  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_32)** **[[Code]](https://github.com/sehyun03/ADA-label-distribution-matching)**

Unsupervised Selective Labeling for More Effective Semi-Supervised Learning  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_25)** **[[Code]](https://github.com/TonyLianLong/UnsupervisedSelectiveLabeling)**

D2ADA: Dynamic Density-Aware Active Domain Adaptation for Semantic Segmentation  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19818-2_26)** **[[Code]](https://github.com/tsunghan-wu/D2ADA)**

When Active Learning Meets Implicit Semantic Data Augmentation  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_25)**

Talisman: Targeted Active Learning for Object Detection with Rare Classes and Slices Using Submodular Mutual Information  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19839-7_1)** **[[Code]](https://github.com/surajkothawade/talisman)**

PT4AL: Using Self-Supervised Pretext Tasks for Active Learning  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_34)** **[[Code]](https://github.com/johnsk95/PT4AL)**

Active learning strategies for weakly-supervised object detection  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_13)** **[[Code]](https://github.com/huyvvo/BiB)**

Active Pointly-Supervised Instance Segmentation  
**[ECCV'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_35)** **[[Code]](https://github.com/chufengt/APIS)**

Contextual Diversity for Active Learning  
**[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_9)** **[[Code]](https://github.com/sharat29ag/CDAL)**

Active Crowd Counting with Limited Supervision  
🕝 **[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_34)**

Weight Decay Scheduling and Knowledge Distillation for Active Learning  
🕝 **[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58574-7_26)**

Consistency-Based Semi-Supervised Active Learning: Towards Minimizing Labeling Cost  
**[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_30)**

Two Stream Active Query Suggestion for Active Learning in Connectomics  
**[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_7)**

Dual Adversarial Network for Deep Active Learning  
**[ECCV'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_40)**

What do I Annotate Next? An Empirical Study of Active Learning for Action Localization  
🕝 **[ECCV'18]** **[[PDF]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Fabian_Caba_What_do_I_ECCV_2018_paper.pdf)**


### IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
Class-Balanced Active Learning for Image Classification  
**[WACV'22]** **[[PDF]](https://openaccess.thecvf.com/content/WACV2022/papers/Bengar_Class-Balanced_Active_Learning_for_Image_Classification_WACV_2022_paper.pdf)** **[[Code]](https://github.com/Javadzb/Class-Balanced-AL)**

Active Adversarial Domain Adaptation  
**[WACV'20]** **[[PDF]](https://openaccess.thecvf.com/content_WACV_2020/papers/Su_Active_Adversarial_Domain_Adaptation_WACV_2020_paper.pdf)**

Region-Based Active Learning for Efficient Labeling in Semantic Segmentation  
**[WACV'19]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8659293)**


### British Machine Vision Conference (BMVC)
CEREALS - Cost-Effective REgion-Based Active Learning for Semantic Segmentation  
**[BMVC'18]** **[[PDF]](http://bmvc2018.org/contents/papers/0437.pdf)**


### International Conference on Learning Representations (ICLR)
A Simple Yet Powerful Deep Active Learning With Snapshots Ensembles  
**[ICLR'23]** **[[PDF]](https://openreview.net/pdf?id=IVESH65r0Ar)** **[[Code]](https://github.com/nannullna/snapshot-al)**

Active Learning for Object Detection with Evidential Deep Learning and Hierarchical Uncertainty Aggregation  
**[ICLR'23]** **[[PDF]](https://openreview.net/pdf?id=MnEjsw-vj-X)** **[[Code]](https://github.com/MoonLab-YH/AOD_MEH_HUA)**

Evidential Uncertainty and Diversity Guided Active Learning for Scene Graph Generation  
**[ICLR'23]** **[[PDF]](https://openreview.net/pdf?id=xI1ZTtVOtlz)**

Dirichlet-Based Uncertainty Calibration for Active Domain Adaptation  
**[ICLR'23]** **[[PDF]](https://openreview.net/pdf?id=4WM4cy42B81)** **[[Code]](https://github.com/BIT-DA/DUC)**

Low-Budget Active Learning via Wasserstein Distance: An Integer Programming Approach  
**[ICLR'22]** **[[PDF]](https://openreview.net/pdf?id=v8OlxjGn23S)**

Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds  
**[ICLR'20]** **[[PDF]](https://openreview.net/pdf?id=ryghZJBKPS)** **[[Code]](https://github.com/JordanAsh/badge)**

Reinforced Active Learning for Image Segmentation  
**[ICLR'20]** **[[PDF]](https://openreview.net/pdf?id=SkgC6TNFvr)** **[[Code]](https://github.com/ArantxaCasanova/ralis)**

Active Learning for Convolutional Neural Networks: A Core-Set Approach  
**[ICLR'18]** **[[PDF]](https://openreview.net/pdf?id=H1aIuk-RW)** **[[Code]](https://github.com/ozansener/active_learning_coreset)**


### Advances in Neural Information Processing Systems (NeurIPS)

Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment  
**[NeurIPS'23]** **[[PDF]](https://openreview.net/pdf?id=Dqn715Txgl)** **[[Code]](https://github.com/IML-DKFZ/realistic-al)**

How to Select Which Active Learning Strategy Is Best Suited for Your Specific Problem and Budget  
🕝 **[NeurIPS'23]** **[[PDF]](https://arxiv.org/pdf/2306.03543.pdf)**  

Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning  
🕝 **[NeurIPS'23]** **[[PDF]](https://openreview.net/pdf?id=lV3LIGlc1w)** **[[Code]](https://github.com/njustkmg/PAL)**

Active Learning for Semantic Segmentation with Multi-class Label Query  
🕝 **[NeurIPS'23]** **[[PDF]](https://arxiv.org/pdf/2309.09319.pdf)** **[[Code]](https://github.com/sehyun03/MulActSeg)**

Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation  
🕝 **[NeurIPS'23]** **[[PDF]](https://arxiv.org/pdf/2310.20293.pdf)** **[[Code]](https://github.com/BIT-DA/Annotator)**

AbdomenAtlas-8K: Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks  
**[NeurIPS'23]** **[[PDF]](https://www.cs.jhu.edu/~alanlab/Pubs23/qu2023abdomenatlas.pdf)** **[[Code]](https://github.com/https://github.com/MrGiovanni/AbdomenAtlas)**

Towards Free Data Selection with General-Purpose Models  
**[NeurIPS'23]** **[[PDF]](https://browse.arxiv.org/pdf/2309.17342.pdf)** **[[Code]](https://github.com/yichen928/FreeSel)**

A Lagrangian Duality Approach to Active Learning  
🕝 **[NeurIPS'22]** **[[PDF]](https://papers.nips.cc/paper_files/paper/2022/file/f475bdd151d8b5fa01215aeda925e75c-Paper-Conference.pdf)**  

Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning  
🕝 **[NeurIPS'22]** **[[PDF]](https://papers.nips.cc/paper_files/paper/2022/file/cba6f4460a1f395f68a88598c86e79bd-Paper-Conference.pdf)**  

Active Learning Helps Pretrained Models Learn the Intended Task  
🕝 **[NeurIPS'22]** **[[PDF]](https://papers.nips.cc/paper_files/paper/2022/file/b43a0e8a35b1c044b18cd843b9771915-Paper-Conference.pdf)**  

Deep Active Learning by Leveraging Training Dynamics  
🕝 **[NeurIPS'22]** **[[PDF]](https://papers.nips.cc/paper_files/paper/2022/file/a102dd5931da01e1b40205490513304c-Paper-Conference.pdf)**

Are all Frames Equal? Active Sparse Labeling for Video Action Detection  
🕝 **[NeurIPS'22]** **[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5c81ea77a383cc2848d721224717fa4b-Paper-Conference.pdf)** **[[Code]](https://github.com/aayushjr/ASL-video)**

Active Learning Through a Covering Lens  
**[NeurIPS'22]** **[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8c64bc3f7796d31caa7c3e6b969bf7da-Paper-Conference.pdf)** **[[Code]](https://github.com/avihu111/TypiClust)**

Gone Fishing: Neural Active Learning with Fisher Embeddings  
**[NeurIPS'21]** **[[PDF]](https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf)** **[[Code]](https://github.com/JordanAsh/badge)**

Batch Active Learning at Scale  
**[NeurIPS'21]** **[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2021/file/64254db8396e404d9223914a0bd355d2-Paper.pdf)**

SIMILAR: Submodular Information Measures Based Active Learning in Realistic Scenarios  
**[NeurIPS'21]** **[[PDF]](https://proceedings.neurips.cc/paper/2021/file/9af08cda54faea9adf40a201794183cf-Paper.pdf)**

Experimental Design for MRI by Greedy Policy Search  
**[NeurIPS'20]** **[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2020/file/daed210307f1dbc6f1dd9551408d999f-Paper.pdf)** **[[Code]](https://github.com/Timsey/pg_mri)**

BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning  
**[NeurIPS'19]** **[[PDF]](https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf)** **[[Code]](https://github.com/BlackHC/BatchBALD)**


### International Conference on Machine Learning (ICML)
Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets  
**[ICML'22]** **[[PDF]](https://proceedings.mlr.press/v162/hacohen22a/hacohen22a.pdf)** **[[Code]](https://github.com/avihu111/TypiClust)**

Bayesian Generative Active Deep Learning  
**[ICML'19]** **[[PDF]](http://proceedings.mlr.press/v97/tran19a/tran19a.pdf)** **[[Code]](https://github.com/toantm/BGADL)**

Deep Bayesian Active Learning with Image Data  
**[ICML'17]** **[[PDF]](http://proceedings.mlr.press/v70/gal17a/gal17a.pdf)**


### AAAI Conference on Artificial Intelligence (AAAI)
Density Matters: Improved Core-Set for Active Domain Adaptive Segmentation  
🕝 **[AAAI'24]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/29308)**

EOAL: Entropic Open-set Active Learning  
🕝 **[AAAI'24]** **[[PDF]](https://arxiv.org/abs/2312.14126)** **[[Code]](https://github.com/bardisafa/EOAL)**

PRISM: A Rich Class of Parameterized Submodular Information Measures for Guided Subset Selection  
**[AAAI'22]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/21264/21013)**

Boosting Active Learning via Improving Test Performance  
**[AAAI'22]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/20834/20593)** **[[Code]](https://github.com/wangt0716/AL-GradNorm)**

Active Learning for Domain Adaptation: An Energy-Based Approach  
**[AAAI'22]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/20850/20609)** **[[Code]](https://github.com/BIT-DA/EADA)**

An Annotation Sparsification Strategy for 3D Medical Image Segmentation via Representative Selection and Self-Training  
**[AAAI'20]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6175/6031)**

Biomedical Image Segmentation via Representative Annotation  
**[AAAI'19]** **[[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/4540/4418)**


### International Joint Conference on Artificial Intelligence (IJCAI)
Deep Active Learning with Adaptive Acquisition  
**[IJCAI'19]** **[[PDF]](https://www.ijcai.org/proceedings/2019/0343.pdf)** **[[Code]](https://github.com/manuelhaussmann/ral)**


### International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)
Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/3963_paper.pdf)**

Adaptive Curriculum Query Strategy for Active Learning in Medical Image Classification  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/1423_paper.pdf)**

Advancing UWF-SLO Vessel Segmentation with Source-Free Active Domain Adaptation and a Novel Multi-Center Dataset  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/2030_paper.pdf)**

An Uncertainty-guided Tiered Self-training Framework for Active Source-free Domain Adaptation in Prostate Segmentation  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/1781_paper.pdf)**

Class-Balancing Deep Active Learning with Auto-Feature Mixing and Minority Push-Pull Sampling  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/2541_paper.pdf)**

Few Slices Suffice: Multi-Faceted Consistency Learning with Active Cross-Annotation for Barely-supervised 3D Medical Image Segmentation  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/0102_paper.pdf)**

FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/0050_paper.pdf)**

OSAL-ND: Open-set Active Learning for Nucleus Detection  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/0661_paper.pdf)**

SBC-AL: Structure and Boundary Consistency-based Active Learning for Medical Image Segmentation  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/3047_paper.pdf)**

The MRI Scanner as a Diagnostic: Image-less Active Sampling  
🕝 **[MICCAI'24]** **[[PDF]](https://papers.miccai.org/miccai-2024/paper/2052_paper.pdf)**

SLPT: Selective Labeling Meets Prompt Tuning on Label-Limited Lesion Segmentation  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_2)**

PLD-AL: Pseudo-label Divergence-Based Active Learning in Carotid Intima-Media Segmentation for Ultrasound Images  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_6)** **[[Code]](https://github.com/CrystalWei626/PLD_AL)**

Adaptive Region Selection for Active Learning in Whole Slide Image Semantic Segmentation  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_9)** **[[Code]](https://github.com/DeepMicroscopy/AdaptiveRegionSelection)**

EdgeAL: An Edge Estimation Based Active Learning Approach for OCT Segmentation  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_8)** **[[Code]](https://github.com/Mak-Ta-Reque/EdgeAL)**

COLosSAL: A Benchmark for Cold-Start Active Learning for 3D Medical Image Segmentation  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_3)** **[[Code]](https://github.com/MedICL-VU/COLosSAL)**

atTRACTive: Semi-automatic white matter tract segmentation using active learning  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_23)** **[[Code]](https://github.com/MIC-DKFZ/atTRACTive_simulations)**

OpenAL: An Efficient Deep Active Learning Framework for Open-Set Pathology Image Classification  
**[MICCAI'23]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_1)** **[[Code]](https://github.com/miccaiif/OpenAL)**

Discrepancy-Based Active Learning for Weakly Supervised Bleeding Segmentation in Wireless Capsule Endoscopy Images  
**[MICCAI'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_3)**

Consistency-Based Semi-Supervised Evidential Active Learning for Diagnostic Radiograph Classification  
**[MICCAI'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_64)**

Warm Start Active Learning with Proxy Labels and Selection via Semi-Supervised Fine-Tuning  
**[MICCAI'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_29)**

Self-Learning and One-Shot Learning Based Single-Slice Annotation for 3D Medical Image Segmentation  
**[MICCAI'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_24)**

Annotation-Efficient Cell Counting  
**[MICCAI'21]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_39)** **[[Code]](https://github.com/cvbmi-research/AnnotationEfficient-CellCounting)**

Partially-Supervised Learning for Vessel Segmentation in Ocular Images  
**[MICCAI'21]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_26)**

Quality-Aware Memory Network for Interactive Volumetric Image Segmentation  
**[MICCAI'21]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_52)** **[[Code]](https://github.com/lingorX/Mem3D)**

Few Is Enough: Task-Augmented Active Meta-learning for Brain Cell Classification  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_36)**

Suggestive Annotation of Brain Tumour Images with Gradient-Guided Sampling  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59719-1_16)**

Attention, Suggestion and Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_1)**

Deep Active Learning for Effective Pulmonary Nodule Detection  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_59)**

Learning Guided Electron Microscopy with Active Acquisition  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59722-1_8)** **[[Code]](https://github.com/lumimim/learning-guided-SEM)**

Active MR K-Space Sampling with Reinforcement Learning  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_3)** **[[Code]](https://github.com/facebookresearch/active-mri-acquisition)**

Deep Active Learning for Breast Cancer Segmentation on Immunohistochemistry Images  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59722-1_49)**

Deep Reinforcement Active Learning for Medical Image Classification  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_4)**

Uncertainty-Guided Efficient Interactive Refinement of Fetal Brain Segmentation from Stacks of MRI Slices  
**[MICCAI'20]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-59719-1_28)** **[[Code]](https://github.com/HiLab-git/UGIR)**

Multiclass Deep Active Learning for Detecting Red Blood Cell Subtypes in Brightfield Microscopy  
**[MICCAI'19]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_76)**

Cost-Sensitive Active Learning for Intracranial Hemorrhage Detection  
**[MICCAI'18]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_82)**

Efficient Active Learning for Image Classification and Segmentation Using a Sample Selection and Conditional Generative Adversarial Network  
**[MICCAI'18]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_65)**

Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation  
**[MICCAI'17]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_46)**


### International Conference on Medical Imaging with Deep Learning (MIDL)
Making Your First Choice: To Address Cold Start Problem in Vision Active Learning  
**[MIDL'23]** **[[PDF]](https://openreview.net/pdf?id=5iSBMWm3ln)** **[[Code]](https://github.com/cliangyu/CSVAL)**

On Learning Adaptive Acquisition Policies for Undersampled Multi-Coil MRI Reconstruction  
**[MIDL'22]** **[[PDF]](https://proceedings.mlr.press/v172/bakker22a/bakker22a.pdf)**

GOAL: Gist-Set Online Active Learning for Efficient Chest X-Ray Image Annotation  
**[MIDL'21]** **[[PDF]](https://proceedings.mlr.press/v143/nguyen21a/nguyen21a.pdf)**


### International Symposium on Biomedical Imaging (ISBI)
Active Learning Enhances Classification of Histopathology Whole Slide Images with Attention-Based Multiple Instance Learning  
**[ISBI'23]** **[[PDF]](https://ieeexplore.ieee.org/document/10230685)**

Rapid Model Transfer for Medical Image Segmentation Via Iterative Human-in-the-Loop Update: from Labelled Public to Unlabelled Clinical Datasets for Multi-Organ Segmentation in CT  
🕝 **[ISBI'22]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9761467)**

Labeling Cost Sensitive Batch Active Learning For Brain Tumor Segmentation  
**[ISBI'21]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9434098)**


### MICCAI Workshops
Active Deep Learning with Fisher Information for Patch-Wise Semantic Segmentation  
**[MICCAIW'18]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_10)**

CLINICAL: Targeted Active Learning for Imbalanced Medical Image Classification  
**[MICCAIW'22]** **[[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-16760-7_12)**


### Annual Meeting of the Association for Computational Linguistics (ACL)
Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering  
**[ACL'21]** **[[PDF]](https://aclanthology.org/2021.acl-long.564.pdf)** **[[Code]](https://github.com/siddk/vqa-outliers)**


### Conference on Empirical Methods in Natural Language Processing (EMNLP)
Cold-Start Active Learning through Self-Supervised Language Modeling  
**[EMNLP'20]** **[[PDF]](https://aclanthology.org/2020.emnlp-main.637.pdf)** **[[Code]](https://github.com/forest-snow/alps)**


### International Conference on Artificial Intelligence and Statistics (AISTATS)
Deep Active Learning: Unified and Principled Method for Query and Training  
**[AISTATS'20]** **[[PDF]](http://proceedings.mlr.press/v108/shui20a/shui20a.pdf)** **[[Code]](https://github.com/cjshui/WAAL)**


### Nature Communications
Active label cleaning for improved dataset quality under resource constraints  
**[Nature Communications 2022]** **[[PDF]](https://www.nature.com/articles/s41467-022-28818-3)** **[[Code]](https://github.com/microsoft/InnerEye-DeepLearning/tree/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality)**  


### npj Digital Medicine
Biological data annotation via a human-augmenting AI-based labeling system  
**[npj Digital Medicine 2021]** **[[PDF]](https://www.nature.com/articles/s41746-021-00520-6)**  


### Scientific Reports
ARA: accurate, reliable and active histopathological image classification framework with Bayesian deep learning  
🕝 **[Scientific Reports 2019]** **[[PDF]](https://www.nature.com/articles/s41598-019-50587-1)**  


### IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
Multiple Instance Differentiation Learning for Active Object Detection  
**[TPAMI 2023]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/10129096)** **[[Code]](https://github.com/WanFang13/MIDL)**

Contrastive Active Learning under Class Distribution Mismatch  
**[TPAMI 2022]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9816025)** **[[Code]](https://github.com/RUC-DWBI-ML/CCAL)**

Active Image Synthesis for Efficient Labeling  
**[TPAMI 2021]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9091327)** **[[Code]](https://github.com/cjshui/WAAL)**

A Probabilistic Active Learning Algorithm Based on Fisher Information Ratio  
**[TPAMI 2018]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8016395)**


### Journal of Machine Learning Research (JMLR)
Asymptotic analysis of objectives based on fisher information in active learning  
**[JMLR 2017]** **[[PDF]](https://www.jmlr.org/papers/volume18/15-104/15-104.pdf)**


### Medical Image Analysis (MedIA)

ALFREDO: Active Learning with FeatuRe disEntangelement and DOmain adaptation for medical image classification  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001865)**  

Reducing annotating load: Active learning with synthetic images in surgical instrument segmentation  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001713)**  

Deep Bayesian active learning-to-rank with relative annotation for estimation of ulcerative colitis severity  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001257)**  

MONAI label: A framework for ai-assisted interactive labeling of 3d medical images  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001324)**  

Active learning using adaptable task-based prioritisation  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524001063)**  

Focused active learning for histopathological image classification  
🕝 **[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841524000872)**  

GANDALF: Graph-based transformer and Data Augmentation Active Learning Framework with interpretable features for multi-label chest Xray classification  
**[MedIA 2024]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841523003353)**

Active learning for medical image segmentation with stochastic batches  
**[MedIA 2023]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841523002189)** **[[Code]](https://github.com/Minimel/StochasticBatchAL)**

HAL-IA: A Hybrid Active Learning framework using Interactive Annotation for medical image segmentation  
**[MedIA 2023]** **[[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001226)**

Deep Active Learning for Suggestive Segmentation of Biomedical Image Stacks via Optimisation of Dice Scores and Traced Boundary Length  
**[MedIA 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841522001967)**

Suggestive Annotation of Brain MR Images with Gradient-Guided Sampling  
**[MedIA 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841522000263)**

Volumetric Memory Network for Interactive Medical Image Segmentation  
**[MedIA 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841522002316)** **[[Code]](https://github.com/lingorX/Mem3D)**

KCB-Net: A 3D Knee Cartilage and Bone Segmentation Network via Sparse Annotation  
**[MedIA 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841522002146)**

COVID-AL: The Diagnosis of COVID-19 with Deep Active Learning  
**[MedIA 2021]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841520302772)**

Active, Continual Fine Tuning of Convolutional Neural Networks for Reducing Annotation Efforts  
**[MedIA 2021]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S1361841521000438)** **[[Code]](https://github.com/MrGiovanni/Active-Learning)**


### IEEE Transactions on Medical Imaging (TMI)
Dual-Reference Source-Free Active Domain Adaptation for Nasopharyngeal Carcinoma Tumor Segmentation across Multiple Hospitals  
**[TMI 2024]** **[[PDF]](https://arxiv.org/pdf/2309.13401.pdf)** **[[Code]](https://github.com/whq-xxh/Active-GTV-Seg)**

Learning from Incorrectness: Active Learning with Negative Pre-training and Curriculum Querying for Histological Tissue Classification  
**[TMI 2023]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/10244066)** **[[Code]](https://github.com/LactorHwt/ICAL)**

Federated Active Learning for Multicenter Collaborative Disease Diagnosis  
**[TMI 2023]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9975277)**

Which Pixel to Annotate: A Label-Efficient Nuclei Segmentation Framework  
**[TMI 2023]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9946007)** **[[Code]](https://github.com/Lewislou/Label-Effcient-Nuclei-Segmentation)**

PathAL: An Active Learning Framework for Histopathology Image Analysis  
**[TMI 2022]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9648200)**

Graph Node Based Interpretability Guided Sample Selection for Active Learning  
**[TMI 2022]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9919866)**

Interpretability-Driven Sample Selection Using Self Supervised Learning for Disease Classification and Segmentation  
**[TMI 2021]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9361645)**

Diminishing Uncertainty within the Training Pool: Active Learning for Medical Image Segmentation  
**[TMI 2021]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9310359)**

Automated Muscle Segmentation from Clinical CT Using Bayesian U-Net for Personalized Musculoskeletal Modeling  
**[TMI 2020]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8830493)**

Rectifying Supporting Regions with Mixed and Active Supervision for Rib Fracture Recognition  
**[TMI 2020]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9130071)**

Intelligent Labeling Based on Fisher Information for Medical Image Segmentation Using Deep Learning  
**[TMI 2019]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8675438)**


### IEEE Journal of Biomedical and Health Informatics (JBHI)
DSAL: Deeply Supervised Active Learning from Strong and Weak Labelers for Biomedical Image Segmentation  
**[JBHI 2021]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/9326423)** **[[Code]](https://github.com/jacobzhaoziyuan/DSAL)**

Label-Efficient Breast Cancer Histopathological Image Classification  
**[JBHI 2019]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8561292)**


### IEEE Transactions on Biomedical Engineering (TBME)
Reliable Label-Efficient Learning for Biomedical Image Recognition  
**[TBME 2018]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/8590767)**


### IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
Cost-Effective Active Learning for Deep Image Classification  
**[TCSVT 2017]** **[[PDF]](https://ieeexplore.ieee.org/abstract/document/7508942)**


### Information Sciences
Cold-Start Active Learning for Image Classification  
**[Information Sciences 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S0020025522011768)**


### Knowledge-Based Systems (KBS)
Deep Active Learning Models for Imbalanced Image Classification  
**[KBS 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S0950705122009248)**

One-Shot Active Learning for Image Segmentation via Contrastive Learning and Diversity-Based Sampling  
**[KBS 2022]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S0950705122000909)**

Active Learning for Segmentation based on Bayesian Sample Queries  
**[KBS 2021]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S0950705120306602)** **[[Code]](https://github.com/firatozdemir/AL-BSQ)**


### Engineering Applications of Artificial Intelligence
Density-Based One-Shot Active Learning for Image Segmentation  
**[Engineering Applications of Artificial Intelligence 2023]** **[[PDF]](https://www.sciencedirect.com/science/article/pii/S0952197623009892)**


### arXiv

Active Self-Training for Weakly Supervised 3D Scene Semantic Segmentation  
**[arXiv 2022]** **[[PDF]](https://arxiv.org/pdf/2209.07069.pdf)** **[[Code]](https://github.com/GengxinLiu/ActiveST)**

Active CT Reconstruction with a Learned Sampling Policy  
**[arXiv 2022]** **[[PDF]](https://arxiv.org/pdf/2211.01670.pdf)**

A simple baseline for low-budget active learning  
**[arxiv 2021]** **[[PDF]](https://arxiv.org/pdf/2110.12033.pdf)** **[[Code]](https://github.com/UCDvision/low-budget-al)**

Self-supervised deep active accelerated MRI  
**[arxiv 2019]** **[[PDF]](https://arxiv.org/pdf/1901.04547.pdf)**

Batch Active Learning Using Determinantal Point Processes  
**[arXiv 2019]** **[[PDF]](https://arxiv.org/pdf/1906.07975.pdf)** **[[Code]](https://github.com/Stanford-ILIAD/DPP-Batch-Active-Learning)**

Discriminative Active Learning  
**[arXiv 2019]** **[[PDF]](https://arxiv.org/pdf/1907.06347.pdf)** **[[Code]](https://github.com/dsgissin/DiscriminativeActiveLearning)**

Adversarial Active Learning for Deep Networks: A Margin Based Approach  
**[arXiv 2018]** **[[PDF]](https://arxiv.org/pdf/1802.09841.pdf)**

Generative Adversarial Active Learning  
**[arXiv 2017]** **[[PDF]](https://arxiv.org/pdf/1702.07956.pdf)** **[[Code]](https://github.com/billzhaox/PyTorch-GAAL)**


## To-do List

- [x] Categorized by venues

- [ ] Categorized by topics or methods

- [x] Add venue, publication time, PDF links

- [x] Add code links (if official implementation available)

- [ ] Add author names

- [ ] **[Ongoing]** Add newly published papers in 2023 (~~MICCAI~~, ~~CVPR~~, ~~ICCV~~, ~~NeurIPS~~, ICLR, ICML)

- [ ] **[Ongoing]** Add newly published papers in 2024 (ICLR)

- [ ] **[Ongoing]** Add related missing papers