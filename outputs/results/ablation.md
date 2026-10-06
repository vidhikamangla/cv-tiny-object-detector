| Row | Change | Params M | Epochs | Train min | ms/img | AP | AP50 | AP75 | AP tiny | AP small | AP med | AR100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | centre-prior assigner, BCE + GIoU, flip only, no EMA | 12.8 | 30 | 50.1 | 3.5 | 9.8 | 20.3 | 8.7 | 4.0 | 14.2 | 24.7 | 20.8 |
| A1 | + mosaic, HSV, scale jitter, EMA | 12.8 | 30 | 42.3 | 3.5 | 13.1 | 25.7 | 11.9 | 5.1 | 19.2 | 31.9 | 26.8 |
| A2 | + TAL assigner, CIoU | 12.8 | 30 | 50.5 | 3.3 | 14.4 | 27.3 | 13.8 | 5.8 | 21.7 | 35.2 | 26.3 |

AP columns are COCO mAP x100. Size bins are ours, not COCO's: tiny < 16^2 px, small 16^2-32^2, med 32^2-96^2 - COCO's `small < 32^2` puts almost all of VisDrone in one bucket and hides the effect.
