GENIUS is designed for universal retrieval over a large and heterogeneous candidate pool. However, our preliminary results suggest that its performance may depend strongly on candidate-pool composition. In particular, candidates from the same dataset may share more similar RQ prefixes and therefore be more difficult to distinguish than candidates from different datasets. We would therefore like to study whether GENIUS remains effective for fine-grained within-dataset retrieval as the candidate pool changes. 

 

Instructions 

 

Please briefly document the completed reproduction pipeline, the main results, and the possible reasons for the remaining gap from the paper. 

 

Then, use the trained full-union model to evaluate COCO text-to-image retrieval under the following candidate-pool settings: 

 

COCO-local pool 

    Use the complete COCO candidate pool only. 

Progressively expanded pools 

    Start from the COCO-local pool and gradually add image candidates from other M-BEIR datasets: 

COCO only; 

COCO + FashionIQ; 

COCO + several additional image datasets; 

the full M-BEIR union pool. 

    This experiment examines how retrieval performance changes as the pool becomes larger and more diverse. 

Controlled candidate-composition experiment 

    Select a fixed subset of COCO test queries and construct candidate pools of the same size. Each pool must contain all relevant COCO targets, while the remaining candidates are sampled as: 

within-dataset distractors: sampled only from COCO; 

cross-dataset distractors: sampled from non-COCO image datasets; 

mixed distractors: 50% from COCO and 50% from other datasets. 

   

Please test at several fixed pool sizes, for example 10K, 50K, and 100K candidates. Use the same queries and the same number of candidates across the three settings, and repeat random sampling with three seeds. Compare whether within-dataset distractors cause a larger performance drop than cross-dataset distractors. 

 

Finally, analyze the RQ identifiers by measuring prefix overlap among COCO candidates at different RQ depths. This can help determine whether stronger within-dataset prefix overlap is associated with poorer retrieval performance. 