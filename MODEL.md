
## Pre-trained model

FCL Long-tailed Model: https://drive.google.com/file/d/144F7srsnVaXFa92dvsQtWm2Sm0b30jpi/view?usp=sharing

### ATL 
ATL model (HICO-DET): https://cloudstor.aarnet.edu.au/plus/s/NfKOuJKV5bUWiIA

ATL model (HOI-COCO)(COCO): https://cloudstor.aarnet.edu.au/plus/s/zZfJM4ctylwAEiZ

ATL model (HOI-COCO)(COCO, HICO): https://cloudstor.aarnet.edu.au/plus/s/zih9Vcdlwbpt92v

### VCL 
Our model will converge at around iteration 500000 in HICO-DET. V-COCO will converge after 200000 iterations. We provide the model parameters that we trained as follows,

VCL on V-COCO: https://drive.google.com/file/d/1SzzMw6fS6fifZkpuar3B40dIl7YLNoYF/view?usp=sharing. I test the result is 47.82. The baseline also decreases compared to the reported result. The model in my reported result is deleted by accident. Empirically, hyper-parameters $lambda_1$ affects V-COCO more apparently.

VCL on HICO: https://drive.google.com/file/d/16unS3joUleoYlweX0iFxlU2cxG8csTQf/view?usp=sharing

VCL on HICO(Res101): https://drive.google.com/file/d/1iiCywBR0gn6n5tPzOvOSmZw_abOmgg53/view?usp=sharing
