# East Avenue Medical Center Diabetic Peripheral Dataset

The dataset (in Microsoft Excel format) is presented as it was finalized by the data collectors. 

The following observations should be made:

## Missing Data
- Patient 36: No values for NS and CAS (%)
- Patient 46: No values for DM_DUR, INS and HBA1C
- Patient 173: No value for NS

## Replacement of Raw Recorded Data
*DM_DUR*
- Values encoded as '<1' is recorded as 1
- Values encoded as '>10' is recorded as 11

*Nerve Conduction Studies*
- Entries encoded as 'NR' (No Response) are recorded as 0
- Entries encoded as 'NO F WAVE' are recorded as 0

## Final Dimensions of the Dataset
The 3 rows with incomplete data were dropped.
The final dimension of the dataset is:
187 patient rows 
41 data columns
