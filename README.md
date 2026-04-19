# **Midterm MLOps**
``` Identity
Name     : Muhamad Nur Rasyid
NRP      : 3324600018
Class    : 2 D4 SDT-A
Lecturer : Renovita Edelani S.Kom, M.Tr.Kom.
Topic    : MLOps Midterm -- Regression
```
---
## **Overview**

### **Tree**
```
MLOps-Midterm
.
├── README.md
├── __init__.py
├── automl_run.py
├── data
│   └── Concrete_Data.xls
├── manuals_run.py
├── requirements.txt
├── results
│   ├── alpha_vs_error.png
│   ├── automl
│   │   ├── metrics.json
│   │   └── null.json
│   ├── logs.json
│   ├── manuals
│   │   ├── logs_20260419_055427.json
│   │   └── null.json
│   ├── pred_vs_actual_auto1.png
│   ├── pred_vs_actual_manual.png
│   ├── residuals_auto1.png
│   └── residuals_manual.png
├── run_pycaret.ps1
├── run_pycaret.sh
├── setup_venv.ps1
├── setup_venv.sh
├── setup_venv_codespace.sh
└── src
    ├── __init__.py
    ├── automl
    │   ├── __init__.py
    │   └── functions.py
    ├── automl_pipeline.py
    ├── evals
    │   ├── __init__.py
    │   └── metrics.py
    ├── logger
    │   ├── __init__.py
    │   └── logger.py
    ├── manuals
    │   ├── __init__.py
    │   └── regression
    │       ├── __init__.py
    │       └── ridge.py
    ├── pipeline.py
    ├── prep
    │   ├── __init__.py
    │   ├── imputer.py
    │   └── norm.py
    ├── tuner.py
    └── visuals
        ├── __init__.py
        └── visualizer.py
```
### **Tentang Data**
Data yang saya gunakan adalah data tentang Kualitas Beton (Concrete) yang dimana ia memiliki 9 kolom (8X, 1y)
``` cols
[
Cement (component 1)(kg in a m^3 mixture),
Blast Furnace Slag (component 2)(kg in a m^3 mixture),
Fly Ash (component 3)(kg in a m^3 mixture),
Water  (component 4)(kg in a m^3 mixture),
Superplasticizer (component 5)(kg in a m^3 mixture),
Coarse Aggregate  (component 6)(kg in a m^3 mixture),
Fine Aggregate (component 7)(kg in a m^3 mixture),
Age (day),
Concrete compressive strength(MPa, megapascals) 
]
```

| Cement (kg/m³) | Blast Furnace Slag (kg/m³) | Fly Ash (kg/m³) | Water (kg/m³) | Superplasticizer (kg/m³) | Coarse Aggregate (kg/m³) | Fine Aggregate (kg/m³) | Age (day) | Compressive Strength (MPa) |
|----------------|----------------------------|-----------------|---------------|--------------------------|--------------------------|------------------------|-----------|-----------------------------|
| 540.0          | 0.0                        | 0.0             | 162.0         | 2.5                      | 1040.0                   | 676.0                  | 28        | 79.99                       |
| 540.0          | 0.0                        | 0.0             | 162.0         | 2.5                      | 1055.0                   | 676.0                  | 28        | 61.89                       |
| 332.5          | 142.5                      | 0.0             | 228.0         | 0.0                      | 932.0                    | 594.0                  | 270       | 40.27                       |
| 332.5          | 142.5                      | 0.0             | 228.0         | 0.0                      | 932.0                    | 594.0                  | 365       | 41.05                       |
| 198.6          | 132.4                      | 0.0             | 192.0         | 0.0                      | 978.4                    | 825.5                  | 360       | 44.30                       |
| 266.0          | 114.0                      | 0.0             | 228.0         | 0.0                      | 932.0                    | 670.0                  | 90        | 47.03                       |
| 380.0          | 95.0                       | 0.0             | 228.0         | 0.0                      | 932.0                    | 594.0                  | 365       | 43.70                       |
| 380.0          | 95.0                       | 0.0             | 228.0         | 0.0                      | 932.0                    | 594.0                  | 28        | 36.45                       |
| 266.0          | 114.0                      | 0.0             | 228.0         | 0.0                      | 932.0                    | 670.0                  | 28        | 45.85                       |
| 475.0          | 0.0                        | 0.0             | 228.0         | 0.0                      | 932.0                    | 594.0                  | 28        | 39.29                       |



### **Algoritma**
Algoritma yang saya gunakan untuk data yang demikian adalah regresi Ridge, sebab semua fitur X berkorelasi untuk menghasilkan output y.
Jadi, secara keseluruhan, step yang dilakukan di sini adalah:
```
[Data] --> [Cleaning] --> [Split] --> [Normalisasi] --> o
o --> [Corr] --> Loop([Model] --> [Metrics] --> [Tuning]) --> Reporting
```
---

## **Cara menjalankan Program (PyCaret)**
``` shell
#1. Clone
git clone https://github.com/nurrasyid14/MLOps-Midterm.git
```
``` bash
# 2a. Run on Linux :
chmod +x setup_venv.sh
chmod +x run_pycaret.sh

./setup_venv.sh
./run_pycaret.sh automl_run.py
```
``` PowerShell 
# Windows
./setup_venv.ps1
./run_pycaret.ps1 automl_run.py
```
## **Cara menjalankan Program Manual**
``` bash
python manuals_run.py
```

---
---

## **Results**

### **AutoML -- PyCaret**


``` json
[
    {
        "run_id": "20260419_221928",
        "best_model": "LGBMRegressor(n_jobs=-1, random_state=42)",
        "metrics": {
            "RMSE": 4.630507066132197,
            "MAE": 3.0779451551443158,
            "R2": 0.9167901843668548
        }
    }
]
```

![AutoML Result](../results/automl/20260419_221928_pred_vs_actual.png)
![AutoML Result](../results/automl/20260419_221928_pred_vs_actual.png)

#### **Analisis**

Dari file log metrics di folder result, (![log](../results/automl/metrics_20260420_021013.json)), kita dapat mendapati berbagai jenis algoritma regresi dan perbandingan metriknya. Di situlah kita mendapat kesimpulan bahwa Pycaret bekerja dengan cara "Run all and Compare"; dan didapat bahwa LGBM Regressor menempati posisi terbaik dengan R^2 0.91--yang bermakna ia dapat menggeneralisasi, dan merepresentasikan 92% dari data. 

Maka, tahap selanjutnya adalah "Build from scratch" yang dimana saya akan mengamati 2 dari keseluruhan algoritma regresi, yakni Ridge, dan LGBM.

---
### **Manual: Ridge & LGBM**

``` json
[
    {
        "model": "ridge",
        "params": {
            "alpha": 0.0001
        },
        "RMSE": 7.462139034114744,
        "MAE": 5.985868194420646,
        "R2": 0.7839052940346638
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.00018329807108324357
        },
        "RMSE": 7.463332436891396,
        "MAE": 5.994384104915397,
        "R2": 0.7838361694464835
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.0003359818286283781
        },
        "RMSE": 7.461365827853951,
        "MAE": 5.99964951584195,
        "R2": 0.7839500739891323
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.0006158482110660267
        },
        "RMSE": 7.456676020092702,
        "MAE": 6.002377624284772,
        "R2": 0.7842215830430512
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.0011288378916846883
        },
        "RMSE": 7.45042700324723,
        "MAE": 6.001140230514239,
        "R2": 0.7845830947806768
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.00206913808111479
        },
        "RMSE": 7.442656524334772,
        "MAE": 5.997966423667234,
        "R2": 0.7850322018236058
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.00379269019073225
        },
        "RMSE": 7.43243373918641,
        "MAE": 5.993341686929292,
        "R2": 0.7856223299271131
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.0069519279617756054
        },
        "RMSE": 7.419338771880634,
        "MAE": 5.984059834791568,
        "R2": 0.7863770747820962
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.012742749857031334
        },
        "RMSE": 7.4039752256746185,
        "MAE": 5.970286213093397,
        "R2": 0.7872608752628083
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.023357214690901212
        },
        "RMSE": 7.38661331067927,
        "MAE": 5.955075830594689,
        "R2": 0.788257428571392
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.04281332398719392
        },
        "RMSE": 7.3666156903584215,
        "MAE": 5.93581148086922,
        "R2": 0.7894023689092793
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.07847599703514607
        },
        "RMSE": 7.345129017037963,
        "MAE": 5.914634105664053,
        "R2": 0.7906291040757605
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.14384498882876628
        },
        "RMSE": 7.328752077249022,
        "MAE": 5.892947615451902,
        "R2": 0.7915617036271243
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.26366508987303583
        },
        "RMSE": 7.330594509396695,
        "MAE": 5.901691206018465,
        "R2": 0.7914568885954318
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.4832930238571752
        },
        "RMSE": 7.369695016469606,
        "MAE": 5.960712211411046,
        "R2": 0.7892262678741675
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 0.8858667904100823
        },
        "RMSE": 7.470057138938592,
        "MAE": 6.087803234386597,
        "R2": 0.7834464528545713
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 1.623776739188721
        },
        "RMSE": 7.650197623337368,
        "MAE": 6.287737765165381,
        "R2": 0.7728761388386753
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 2.9763514416313193
        },
        "RMSE": 7.906522051518919,
        "MAE": 6.522252653882861,
        "R2": 0.7574013234203758
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 5.455594781168514
        },
        "RMSE": 8.221576881768094,
        "MAE": 6.772775421791403,
        "R2": 0.7376822376727292
    },
    {
        "model": "ridge",
        "params": {
            "alpha": 10.0
        },
        "RMSE": 8.596491932901888,
        "MAE": 7.062574857883111,
        "R2": 0.7132126610488978
    },
    {
        "model": "lgbm",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1
        },
        "RMSE": 4.309117765862747,
        "MAE": 2.9557459651397964,
        "R2": 0.9279400127598255
    },
    {
        "model": "lgbm",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05
        },
        "RMSE": 4.342085956432768,
        "MAE": 2.9797279401199965,
        "R2": 0.9268331619036506
    }
]
```
- Ridge
![Ridge Result](../results/manuals/20260420_000032_ridge_pred_vs_actual.png)
![Ridge Residual](../results/manuals/20260420_000032_ridge_residuals.png)
![Alpha v Error](../results/manuals/20260420_000032_ridge_alpha_vs_error.png)
- LGBM
![LGBM Result](../results/manuals/20260420_000032_lgbm_pred_vs_actual.png)
![LGBM Residual](../results/manuals/20260420_000032_lgbm_residuals.png)

**Output Terminal**
``` log
Model: ridge
RMSE: 7.3288
MAE : 5.8929
R2  : 0.7916

Model: lgbm
RMSE: 4.3091
MAE : 2.9557
R2  : 0.9279

Best Alpha (Ridge): 0.14384498882876628

=== LEADERBOARD ===
1. lgbm → RMSE: 4.3091
2. ridge → RMSE: 7.3288
```
#### **Analisis**
Sedikit lebih baik dari PyCaret, dimana di sini saya dapat mendapatkan R^2 sebesar 0.93 dalam LGBM, dan 0.79 dalam Ridge; yang saya lakukan persis dengan PyCaret, namum saya hanya menggunakan dua diantaranya. Adapun mengenai alpha pada ridge, saya menyatakan demikian: " alphas = np.logspace(-4, 1, 20)" yang maknanya adalah range dari alpha adalah:
``` LaTeX
# Fungsi Menghasilkan 20 nilai alpha secara logaritmik dari 10^{-4} hingga 10^{1}, dengan formula
# \alpha_i = 10^{\, -4 + i \cdot \frac{1 - (-4)}{19}} = 10^{\, -4 + i \cdot \frac{5}{19}},
# untuk i = 0, 1, \ldots, 19
```
---
## **Analisis Keseluruhan**

PyCaret akan sangat berguna swbagai alat "overview", karena ia akan memilihkan algoritma terbaik, dan dengan outputyang dihasilkan dari data sampel, kita dapat mengetahui algoritma apa yang paling cocok digunakan untuk mengatasi jenis data tersebut. Sehingga dengan gambaran yang diberikan PyCaret, kita dapat membangun pipeline yang lebih mutakhir dalam mengatasi data tersebut. 
## **Detail Presentasi**
``` sequence
[Title] - [Overview Tugas] - [Keterangan Data] - [Overview Alur Data] - o
o -[FLow PyCaret] - [Tree] -  [Flow Manual] - [Output] - [Comparison] - [Kesimpulan]
```
