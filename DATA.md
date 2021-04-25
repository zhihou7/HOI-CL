### Data
We present the differences between different detector in our paper and analyze the effect of object boxes on HOI detection. VCL detector and DRG detector can be download from the corresponding paper. Due to the space limitation of Google Drive, there are many files provided in CloudStor. Many thanks to CloudStor and The University of Sydney.
Here, we provide the GT boxes.

GT boxes annotation: https://drive.google.com/file/d/15UXbsoverISJ9wNO-84uI4kQEbRjyRa8/view?usp=sharing

FCL was finished about 10 months ago. In the first submission, we compare the difference among COCO detector, Fine-tuned Detector and GT boxes. We further find DRG object detector largely increases the baseline. 
All these comparisons illustrate the significant effect of object detector on HOI. That's really necessary to provide the performance of object detector.

HOI-COCO training data: https://cloudstor.aarnet.edu.au/plus/s/6NzReMWHblQVpht

Please notice train2017 might contain part of V-COCO test data. Thus, we just use train2014 in our experiment. If we use train2017, the result might be better (improve about 0.5%). We think that is the case: we have localized objects, but we do not know the interaction. 

#### Affordance Recognition evaltion dataset

Evalation of Object365_COCO: https://cloudstor.aarnet.edu.au/plus/s/GpNkjOHaS8xN5ar

Evalation of COCO (val2017): https://cloudstor.aarnet.edu.au/plus/s/QBXOYiJ5NHqtcti

Evalation of Object365 (novel classes): https://cloudstor.aarnet.edu.au/plus/s/pRBLkhy9TUm5xGo

Evalation of HICO (test): https://cloudstor.aarnet.edu.au/plus/s/AFrv822lPC30iHt



#### Object Dataset (HICO)
Here we provide the object datasets that we use in this repo. The format is the same as the training data of HICO.

COCO: https://cloudstor.aarnet.edu.au/plus/s/9cgMYKq5B4waawA

#### Object Dataset (HOI-COCO)
Object365_COCO: https://cloudstor.aarnet.edu.au/plus/s/VuHvDdp5msRnpqn. (we do not run this experiment).

COCO: https://cloudstor.aarnet.edu.au/plus/s/N35ovTXWtLmG9ZN

HOI-COCO: https://cloudstor.aarnet.edu.au/plus/s/YEiPiX0B3jaFasU
