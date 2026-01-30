@echo off
chcp 65001 >nul
cls
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║         ENTRAINEMENT MOBILENET - DETECTION DE MORPHING              ║
echo ║                     Deep Learning Detector                           ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo    Configuration recommandee:
echo.
echo    Dataset:
echo      - Morphs:     morphing_results (1124 images)
echo      - Bona fide:  sample_data/before_morph (20 images)
echo.
echo    Parametres:
echo      - Epochs:       50 (defaut) ou 100 (meilleure precision)
echo      - Batch size:   32 (defaut) ou 16 (si memoire limitee)
echo      - Image size:   224x224 (defaut MobileNet)
echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
echo    Choisissez une configuration:
echo.
echo    [1] RAPIDE - 20 epochs (test rapide, ~5-10 min)
echo    [2] STANDARD - 50 epochs (recommande, ~15-20 min)
echo    [3] AVANCEE - 100 epochs (meilleure precision, ~30-40 min)
echo    [4] PERSONNALISEE (choisir les parametres)
echo.
echo    [0] QUITTER
echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
set /p choice="Votre choix (1-4 ou 0): "

if "%choice%"=="1" goto rapide
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto avancee
if "%choice%"=="4" goto personnalisee
if "%choice%"=="0" goto fin
goto invalide

:rapide
cls
echo ══════════════════════════════════════════════════════════════════════
echo                       CONFIGURATION RAPIDE
echo                         20 epochs - Test
echo ══════════════════════════════════════════════════════════════════════
echo.
python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --epochs 20 --batch-size 32
goto resultats

:standard
cls
echo ══════════════════════════════════════════════════════════════════════
echo                      CONFIGURATION STANDARD
echo                     50 epochs - Recommandee
echo ══════════════════════════════════════════════════════════════════════
echo.
python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --epochs 50 --batch-size 32
goto resultats

:avancee
cls
echo ══════════════════════════════════════════════════════════════════════
echo                      CONFIGURATION AVANCEE
echo                  100 epochs - Meilleure Precision
echo ══════════════════════════════════════════════════════════════════════
echo.
python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --epochs 100 --batch-size 32
goto resultats

:personnalisee
cls
echo ══════════════════════════════════════════════════════════════════════
echo                    CONFIGURATION PERSONNALISEE
echo ══════════════════════════════════════════════════════════════════════
echo.
set /p epochs="Nombre d'epochs (defaut 50): "
set /p batch="Batch size (defaut 32): "
set /p imgsize="Image size (defaut 224): "

if "%epochs%"=="" set epochs=50
if "%batch%"=="" set batch=32
if "%imgsize%"=="" set imgsize=224

echo.
echo Configuration:
echo   Epochs:      %epochs%
echo   Batch size:  %batch%
echo   Image size:  %imgsize%x%imgsize%
echo.
pause
python train_mobilenet_detector.py --morph morphing_results --bona-fide sample_data/before_morph --epochs %epochs% --batch-size %batch% --img-size %imgsize%
goto resultats

:resultats
echo.
echo ══════════════════════════════════════════════════════════════════════
echo.
echo ENTRAINEMENT TERMINE!
echo.
echo Resultats sauvegardes dans: model_output\
echo.
echo   Dossiers crees:
echo     - model_output\models\         (Modeles entraines .keras)
echo     - model_output\plots\          (Graphiques de performance)
echo     - model_output\logs\           (Logs TensorBoard)
echo.
echo   Visualisations generees:
echo     - training_curves.png          (Courbes d'entrainement)
echo     - confusion_matrix.png         (Matrice de confusion)
echo     - roc_curve.png                (Courbe ROC)
echo     - precision_recall_curve.png   (Courbe Precision-Recall)
echo.
pause
echo.
echo Voulez-vous ouvrir le dossier des resultats? (O/N)
set /p open_results="> "
if /i "%open_results%"=="O" explorer model_output
goto menu

:invalide
echo.
echo Choix invalide! Veuillez choisir 1, 2, 3, 4 ou 0.
pause
goto :eof

:menu
cls
goto :eof

:fin
echo.
echo Au revoir!
timeout /t 2 >nul
exit
