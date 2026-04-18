# **Midterm MLOps**
``` Identity
Name     : Muhamad Nur Rasyid
NRP      : 3324600018
Class    : 2 D4 SDT-A
Lecturer : Renovita Edelani S.Kom, M.Tr.Kom.
Topic    : MLOps Midterm -- Regression
```
## **Overview**
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
### **Algoritma**
Algoritma yang saya gunakan untuk data yang demikian adalah regresi Ridge, sebab semua fitur X berkorelasi untuk menghasilkan output y.
Jadi, secara keseluruhan, step yang dilakukan di sini adalah:
```
[Data] --> [Cleaning] --> [Split] --> [Normalisasi] --> o
o --> [Corr] --> Loop([Model] --> [Metrics] --> [Tuning]) --> Reporting
```

## **Cara menjalankan Program (PyCaret)**
``` shell
git clone https://github.com/nurrasyid14/MLOps-Midterm.git
# Linux :
chmod +x setup_venv.sh
chmod +x run_pycaret.sh

./setup_venv.sh
./run_pycaret.sh automl_run.py

# Windows
./setup_venv.ps1
./run_pycaret.ps1 automl_run.py
```

## **Results**

### **AutoML -- PyCaret**
``` json
[
    {
        "alpha": "LGBMRegressor(n_jobs=-1, random_state=42)",
        "RMSE": 4.630507066132197,
        "MAE": 3.0779451551443158,
        "R2": 0.9167901843668548
    }
]
```
![AutoML Result](results/pred_vs_actual_auto1.png)
![AutoML Result](results/residuals_auto1.png)

