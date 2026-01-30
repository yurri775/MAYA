@echo off
chcp 65001 > nul
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo.
echo        📊 GÉNÉRATION DU DATASET DE VISAGES MORPHÉS 📊
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo.
echo Ce script va générer:
echo   • K identités morphées (combinaisons de 2 personnes)
echo   • 30 images par identité morphée
echo   • Mélange fixé à 50%% (alpha = 0.5)
echo   • Format de nommage: A_B_N
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo.
pause

python generate_morphed_dataset.py

echo.
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo.
echo                    ✅ TERMINÉ! ✅
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo.
pause
