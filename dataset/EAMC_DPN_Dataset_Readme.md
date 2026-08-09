# Diabetic Peripheral Neuropathy Dataset - East Avenue Medical Center 

## About the Dataset
The dataset (in Microsoft Excel format) is presented as it was finalized by the data collectors. The data was collected from October 2018 up to July 2023 at the East Avenue Medical Center (EAMC). This study was approved by the Ethics Committee of EAMC and all subjects gave their written consent prior to participating. For more details about this study, inquiries can be sent to <insert here an EAMC institutional email address that will handle requests for this dataset>.

## Data Columns

- CODE
    - Patient code (numerically increasing values from 1 to 190)


**PROFILE**

*Patient Profile*
- SEX
    - Patient's sex 
    - binary (M or F)
- AGE
    - Patient's age
    - continuous
- SUBJ
    - Other symptoms 
    - binary (Y or N)
- DM DUR
    - Diabetes Mellitus duration 
    - continuous (years)
- INSULIN
    - Patient is taking insulin treatments 
    - binary (Y or N)
- HBA1C
    - Hemoglobin A1c
    - continuous
- DATE
    - Date the data was taken


**COMORBIDS**

*Presence of Comorbidities*
- HPN
    - Hypertension
    - binary (Y or N)
- PAOD
    - Peripheral Arterial Occulsive Disease
    - binary (Y or N)
- DSLPDMIA
    - Dyslipidemia
    - binary (Y or N)
- CKD
    - Chronic Kidney Disease
    - binary (Y or N)
- GBS
    - Guillain-Barre Syndrome
    - binary (Y or N)

**NEURO EXAM**

*Values from a Neurology Examination*
- DEC VS
    - Decreased Vibration Sensation
    - binary (Y or N)
- DEC PPS
    - Decreased Pinprick Sensation
    - binary (Y or N)
- DEC LTS
    - Decreased Light Touch Sensation
    - binary (Y or N)
- DEC AR
    - Decreased Ankle Reflex
    - binary (Y or N)


**MNSI**
- MNSI
    - Michigan Neuropathy Screening Instrument
    - continuous


**NERVE CONDUCTION STUDIES**

- LEFT
    - *Results for LEFT Nerve Conduction Studies*
    - SURAL SAP
        - Uv
            - Left Sural Sap Nerve Sensory Amplitude 
            - continuous (Uv)
        - m/s
            - Left Sural Nerve Sensory Conduction 
            - continuous (m/s)
    - SUPER PERONEAL SAP
        - Uv
            - Left Superficial Peroneal Nerve Sensory Amplitude 
            - continuous (Uv)
        - m/s
            - Left Superficial Peroneal Sensory Conductivity Velocity 
            - continuous (m/s)
    - POSTERIOR TIBIAL
        - MCV
            - Left Posterior Tibial Nerve Conduction Velocity
            - continuous
        - DL
            - Left Posterior Tibial Nerve Distal Latency
            - continuous
        - CMAP-ANK
            - Left Ankle Compound Muscle Action Potential
            - continuous
        - CMAP-KNE
            - Left Knee Compound Muscle Action Potential
            - continuous
        - F WAVE
            - Left F-wave latency
            - continuous

- RIGHT
    - *Results for RIGHT Nerve Conduction Studies*
    - SURAL SAP
        - Uv
            - Right Sural Sap Nerve Sensory Amplitude 
            - continuous (Uv)
        - m/s
            - Right Sural Nerve Sensory Conduction 
            - continuous (m/s)
    - SUPER PERONEAL SAP
        - Uv
            - Right Superficial Peroneal Nerve Sensory Amplitude 
            - continuous (Uv)
        - m/s
            - Right Superficial Peroneal Sensory Conductivity Velocity 
            - continuous (m/s)
    - POSTERIOR TIBIAL
        - MCV
            - Right Posterior Tibial Nerve Conduction Velocity
            - continuous
        - DL
            - Right Posterior Tibial Nerve Distal Latency
            - continuous
        - CMAP-ANK
            - Right Ankle Compound Muscle Action Potential
            - continuous
        - CMAP-KNE
            - Right Knee Compound Muscle Action Potential
            - continuous
        - F WAVE
            - Right F-wave latency
            - continuous

**SUDOSCAN**

*Sudoscan screening values*
- FEET
    - MEAN ESC
        - Feet Mean Electrochemical Skin Conductance
        - continuous
    - (%)ASSYM
        - Feet Percentage of Asymmetry 
        - continuous (%)
- HANDS
    - MEAN ESC
        - Hands Mean Electrochemical Skin Conductance
        - continuous
    - (%)ASSYM
        - Hands Percentage of Asymmetry 
        - continuous (%)
- NS
    - Neuropathy Score
    - continuous
- CAS
    - Cardiac Autonomic Neuropathy Score 
    - continuous (%)

**DPN CLASSIFICATION**

*DPN classification for each patient*
- Confirmed	
- Probable	
- Possible	
- Any DPN


## Suggested pre-processing 
Before feeding to Machine Learning models the following pre-processing should be done:

### Missing Data
The following 3 rows with incomplete data can be dropped.
- Patient 36: No values for NS and CAS (%)
- Patient 46: No values for DM_DUR, INS and HBA1C
- Patient 173: No value for NS

### Replacement of Raw Recorded Data
- DM_DUR
    - Values encoded as '<1' can be recorded as 1
    - Values encoded as '>10' can be recorded as 11

- Nerve Conduction Studies
    - Entries encoded as 'NR' (No Response) can be recorded as 0
    - Entries encoded as 'NO F WAVE' can be recorded as 0

After executing the above suggestions, the final dimension of the dataset is:
- 187 patient rows
- 41 data columns


## Publications
Future publications that refer to this dataset will be listed here.

## Data Use Agreement
The use of this dataset is governed by a Data Use Agreement between the East Avenue Medical Center and the person or institution who has been granted permission to access and use this dataset for the specified purpose.  Strict adherence to this agreement must be observed at all times.

## Last Updated
This documentation was last updated August 9, 2026.
