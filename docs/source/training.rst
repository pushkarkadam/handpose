========
Training
========

Training is performed by using
:func:`handpose.train.train_model`.

The following flowchart can be used to visualise th flow fo data.


.. mermaid::
    
    flowchart TD
        Z[CNN] -- outputs --> A[network_head] --pred_head --> B[head_activations]
        B -- pred_head_act --> C[best_box]
        C -- best_pred_head --> D[extract_head]
        D -- pred_data --> E[non_max_suppression]
        D -- pred_data --> F[mean_average_precision]
        E -- pred_data_nms --> F
